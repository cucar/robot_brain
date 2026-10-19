# Universal Compression with Actions and Rewards (UCAR)

UCAR is an algorithm that compresses what it observes and acts on the best reward estimate it holds. A machine
running it is a **UCAR machine**, and like a Turing machine it is defined by two alphabets: the **event alphabet**
it can observe and the **action alphabet** it can execute. Every symbol, base or learned, event or action, is a
**neuron**.

Its inputs are the events observed, the actions that ran and the rewards earned. Its output is what it places
at the frame ahead, written in the base alphabet: the actions and the events it infers.

Above the alphabets it builds a hierarchical dictionary of **patterns**. A pattern is a variable or a function.
A **variable** is what changes from one occurrence of a pattern to the next, and is left open until the
occurrence fills it. A **function** names neurons and variables, each at an offset from the others, and holds
wherever they stand so. A function has a single child. A variable has multiple children: one child per value.
Each frame, the machine compresses what arrived one level at a time. The neurons that fired bid their
patterns, an election takes the bids that pay, and each accepted pattern's child fires at the next level.

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
base symbols or the children of other patterns. This substitution is the whole mechanism for compression.

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
named by a pattern. The pattern's child then stands on the apex in its place, and holds connections of its own.
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

> **D35 — Reward.** A frame carries an array of rewards for actions already executed, each
> ```
> { reward, channels, frames }
> ```
> `reward` is a signed number. `channels` is the set of action channels it pays. `frames` is the span of
> frames it pays over, counted back from the frame it arrives in. Both are optional, and an omitted one means
> all of them.

A reward is an input, not a symbol: nothing fires for it and no pattern names it. Zero is neutral, a positive
reward is good and a negative one is bad. The numbers are the world's to choose. The only condition is that
they are consistent with one another, a larger reward always meaning a better outcome, since the machine only
ever compares them.

A reward is **scoped** when it names channels, a span, or both, and **unscoped** when it names neither. When the
world knows what it is paying for, it scopes the reward. Example: a stock channel, and the frames a position was
held. When the world does not know, the reward pays every action channel over the longest span any neuron is
active for. A scoped reward reaches only the base actions of the channels it names.

# 3. Machine

The machine is a processor with its own memory, running the procedure of Parts III and V once per frame, and a
population of **neurons**, each a processor with a memory of its own, running the procedure of Parts II and
IV when called. The neurons are the base symbols the world declares and the interneurons the machine builds
above them, and the machine reaches each by its id (D2), a base neuron also by the base symbol it was mapped to
(D52), and an interneuron also by what it stands for (D36). A neuron fires as an **activation**, at a frame and a
position. The machine's memory holds what spans the neurons: the three indexes, the open activations, the
activations no pattern covers, which form the **apex**, its state at a frame, the coverage set, and the standing
inferences. Each
frame is one wave through the neurons: it rises through the frame as recognition, runs forward into the frame
ahead as inference, and comes down to the base as expansion, where one winner per dimension is output. What
the machine holds is priced as a **file**, the dictionary of every neuron's patterns and the body the apex
writes, and shortening that file is the first objective.

## 3.1 Neurons

> **D2 — Neuron.** A neuron is a symbol, and a type: its activations are its instances (D8). It is identified
> by its id. Ids are handed out in sequence, so a smaller id is an older neuron. It is a processor with a
> memory of its own, and that memory holds a patterns table (D32), a history (D18) and connections (D25). It
> is a base neuron (D52) or an interneuron (D53).

Older neurons have smaller ids. That's how ties are broken.

The machine holds an index of its neurons by id. Every line, connection, activation and deletion that names a
neuron names it by id, and reaches it through this index.

> **D11 — Identity.** A neuron's identity is its type alone, a base symbol or an interneuron's id (D2), so
> nothing about where it occurred is part of it. **The same shape at two positions is two activations of one neuron**,
> and they pool: both carry the same relative neighborhood, so the same patterns cover them and one pattern
> serves both. A shape learned anywhere is learned everywhere, and the dictionary holds it once.

## 3.2 Base neurons

> **D52 — Base neuron.** A symbol of the alphabet: an event or an action of one dimension of one channel (D1,
> D3). At declaration the machine maps each base symbol `(dimension, bucket)` to a base neuron, in declaration
> order, and keeps the map. The base neurons are the first neurons the machine holds.

Neuron dimensions belong to the base alphabet and to nothing above it.

The map is the machine's index of its base neurons by base symbol: a symbol a frame reports reaches the neuron
that fires through it (D8), and expansion ends on base symbols read back through it (R28).

## 3.3 Interneurons

> **D53 — Interneuron.** A pattern's child: a call neuron (D48) or a value neuron (D45). The machine creates it
> the first time a bid carrying its pattern is accepted and no interneuron already stands for the same thing
> (R16, R43, D36). It has an id and no dimension and no channel.
> Its activation fires over what its bid covered (D31), and the machine places it: **in time at the latest
> frame covered**, since an activation stands in no frame before what it covers has happened (D5); **in every
> other activation dimension at the center of the box the covered nodes span**, rounded toward the smaller
> coordinate on a half, which is where its span is least (D4). A call neuron is placed so over what its function
> covered, a lone parameter's value neuron over the parameter's positions, and a class's value neuron stands
> where its member stood (D45).

An interneuron is not an event or an action, though what it stands for may be either. A child may have
several parents, since several patterns may come to share it (R43), and they may be of either kind. Its
activation's coordinate is within the box its bid covers, at the box's end in time and its middle in space.
Time and space are placed by different rules because the future has not happened: in time the end of what is
covered is the latest point the activation can see, and in space nothing is ahead or behind.

> **D36 — Interneuron index.** The machine's index of its interneurons by what each stands for, beside the one
> by id (D2) and the one by base symbol (D52). An interneuron stands for its expansion (R28): a call neuron for
> its function's relations (D34) with the variables left open, a value neuron for its variable with its member,
> each node in them for what it stands for in turn, down to base symbols. No line names a neuron built over its
> own child (D27), so what a neuron stands for never contains itself and is finite. **No two interneurons
> stand for the same thing.** An accepted bid whose pattern has no child is wired to the interneuron that stands
> for what its child would, whether that neuron bid this frame, fired frames ago or has not fired within any
> open window, and a new one is created only where none does (R16, R43). A line that re-centers changes what its
> child stands for (D29), and where that is what another interneuron stands for, the line is wired to that
> neuron: merging, across owners (D43).

**A pattern is a pointer to a child, and a child is a neuron.** The two are made apart. A neuron adds a
**pattern**, a line in its own table that may enter a cover at once, and bids it with no child. The first
time a bid of that pattern is accepted the machine gives it a **child**, a neuron (D2): one it mints, or
one another pattern already has for the same activations (R16, R43). **Lines stay with the neuron, and neurons belong
to the machine.**

> **D48 — Call neuron.** A function's child, one per function: the neuron the machine gives a function
> the first time a bid carrying it is accepted (R16), shared by every function the machine has wired to it
> (R43). It fires when a bid carrying the function is accepted, over what the bid covered (D53), once, covering
> what the function names, the bidder among them where the function names the owner (§6.4), and the value
> neurons of the variables the function names fire where each value stood (D45). It says which function ran, and its
> connections are one set for every value those variables have taken, so a neuron above learns about the
> function regardless of its values by a pattern over it alone. It holds a table, a history and connections of
> its own, and is expanded through the function whose bid fired it (R28).

> **D45 — Value neuron.** A variable's child, one per value: per class and member, and per parameter and
> member, shared by every variable the machine has wired to it (R43). The owner's table records what each
> one stands for, and that is what it is as a neighbor: **its variable, with its value**, a class's member where
> it stood, or a parameter's member at each of its positions (D7). The value at a position is the neuron that stood there, followed through value neurons
> to a neuron that is not one, so a value neuron standing for a value neuron stands for what that one stands
> for. Two variables of one owner that hold the same neuron in one activation fire two value neurons, and a
> variable that holds the same value again fires the same one. A value neuron has an id like any
> child (D2), and holds a table, a history and connections of its own.
>
> **It is the one neuron its variable fires.** It fires when a bid carrying its variable with that value is
> accepted (§6.4), a function's bid or the parameter's own, where the value stood, once per relation naming the
> variable: a class's value neuron where the member stood, and a parameter's over its positions, placed as a
> child is (D53), however many positions it holds. In a function's bid it stands beside the function's child: the child says which function
> ran, and the value neurons say with what. The child's connections are one set for every value, and a value
> neuron holds connections of its own for its value, so a neuron above can learn about the function regardless
> of its values by a pattern over the child, and about one value where that matters by a pattern over the child
> and the value neuron together. Bid alone, which only a parameter is (D31), the value neuron is all that fires:
> the parameter, with that value. It costs what its variable's choice cost to write, and the whole bid where the
> parameter bid alone (D13), and it expands to its value at the positions of its parameter (R28).

## 3.4 Activations

> **D8 — Activation.** A neuron is activated (fires) only when something happens: an event neuron when its event
> is observed, or an action neuron when its action runs.
>
> **A neuron may fire many times in one frame**, and those activations are instances of one type.
>
> **At the base, at most one neuron fires per dimension per position in a frame.** The input reports one symbol
> per channel at each point of its layout, and a dimension with nothing to report there is silent. That bound
> is a property of the input, not a rule the machine imposes.
>
> **Above the base, multiple neurons can fire for the same dimension and position.** A neuron can choose to
> cover with multiple patterns (D28), so several of its children, and the value neurons beside them (D45), may be
> active at one coordinate, each placed by what its bid covered (D53).

> **D9 — Activation window.** An activation at frame `f` stays open through `f + reach_t`, its reach
> in time (D4). Two things reach it over that span, one frame at a time, by the connect call after the
> frame's last level (§7.1):
>
> | Input           | Description                                                                                                 | References |
> |-----------------|-------------------------------------------------------------------------------------------------------------|------------|
> | the apex       | What stands on the apex of that frame within reach (R27, D25), while the activation is uncovered; in the frame it is covered, the neuron that covered it (D10). | D25, R31   |
> | the rewards     | For the actions of that frame, and of earlier frames the reward spans, each at the distance its frame names. | R29, R33   |
>
> The machine holds the activation open, not the neuron:
> ```
> open activation  =  (the activation, its coordinate, its age, covered at)      one per (neuron, age, position)
> ```
>
> **Age is per activation, and a neuron can carry several ages at once.** An activation's age is the frames
> elapsed since the frame of its coordinate, `0` through `reach_t`; a new activation is the one at age 0. A
> child fired at a coordinate frames back (D53) opens at the age that coordinate gives it, and its first
> connect call hands it the frames it was not there for (§7.1).
>
> **Covered at** is the age at which an accepted bid first covered the activation, none until one does.
> Coverage is acquired late and never revoked (R27), so this one number says for every frame of the window
> whether the activation stood on the apex there: it did at every age before it, and at none from it on.
>
> **An open activation carries no commitment beyond that one number.** It holds its neighborhood, the age it
> was covered at, and nothing else, because everything else it was going to decide was decided at age 0 (R1).
> It infers once, at age 0, and the machine keeps what it inferred (R45); what it does for the rest of its life
> is connect to what ran, and that is Part IV.

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
> **A child is as strong as what it covers.** An accepted bid's child is strong when any activation it covers
> is strong, and weak when every one of them is: a situation made entirely of inferred inputs is inferred. A
> value neuron is as strong as the value it stands for (D45).

> **D37 — Call.** An activation of a call neuron (D48), with one value neuron beside it per variable its
> function names (D45). It is made in one of two ways and is the same activation either way. **Recognized**: the
> election accepts a bid carrying the function, and the call fires over what the bid covered (D53), as strong as what
> it covers (D40). **Inferred**: an expansion places the call neuron at its completion, the voter's coordinate
> plus the connection's offset in time and in every other activation dimension (R28, R36), and the call fires
> weakly there, for the report to make strong what it placed (R30). A call carries nothing of its own: what it is
> stands in the dictionary line, and what its variables held stands in the value neurons beside it. Expanding it
> recovers its arguments, a parameter's member at every one of its positions and a class's member where the
> function named it; an
> inferred call with no value neuron beside a parameter binds it from what stands at those of its positions that
> have happened and places that at the rest (D44).

**A pattern is matched or inferred, and both kinds of member take part in both.** Matched, a pattern says that
its events stood and its actions ran. Inferred, it is expanded (R28): its actions are placed as the machine's
output and its events beside them, both weakly, and the report makes strong what ran and what was seen. What
follows any activation is held on its neuron as connections (D25), one kind, and an inference is a connection
read.

## 3.5 Space

> **D4 — Reach.** The farthest an activation sees, in each activation dimension (D1): its span plus the radius
> `R` of that dimension, in positions.
> ```
> span(a base activation)    =   0
> span(an activation above)  =   the largest, over everything the pattern that fired it names,
>                                of that node's offset from the activation plus the span of what stood there
> reach                      =   span + R                                        per activation dimension
> ```
> The span is how far the farthest base activation under the activation lies from its coordinate, read from
> where the nodes its pattern names stood when its bid fired it (D31). A pattern may stand at a different
> distance from its nodes on each occurrence, so the span is the activation's, and it is fixed when the
> activation fires. In every dimension but time the activation stands at the center of what it covers (D53), so
> its span there is half the box. An activation sees `R` beyond the farthest node it stands for. A base
> activation stands for nothing but itself and sees `R`; a child over a pair of adjacent positions sees `R + 1`; a child whose
> coverage ends where another's begins sees that child's coordinate, one past its span, which is how two
> adjacent coverages are joined into one by a child above them. A child one letter longer than its parent's
> sees one position further. In space the reach runs either way; in time it runs back, and the same number
> runs forward as the activation's window (D9). The reach is a bound, not a distance: an activation sees every
> distance up to it, exact within `R` and rounded to one significant digit beyond when written (D6). `reach_t`
> is this reach in the time dimension, the activation's window.

> **D6 — Offsets.** The offset between two activations is the difference of their coordinates, one component per
> activation dimension they share (D1, D2). A neighborhood keeps it exact (D18). Written, in a line (D15), in a
> connection (D25) or where an expansion places a neighbor (R28), each component is **rounded down to one
> significant digit in base `R + 1`**, `R` being the radius of that dimension (D1):
> ```
> offset(0)   =   0
> offset(x)   =   sign(x) · floor(|x| / (R + 1)^k) · (R + 1)^k         k = floor(log_(R + 1) |x|)
> ```
> The single digits of base `R + 1` are `0` through `R`, so every distance up to `R` is exact, and beyond it the
> distance is kept to one digit. At `R = 1`, 5 and 7 become 4 and −13 becomes −8; at `R = 10`, 7 stays 7, 15
> becomes 11, 95 becomes 88 and −250 becomes −242. The reachable offsets are `0`, every distance up to `±R`, then
> `±(R + 1), ±2(R + 1), … ±R(R + 1), ±(R + 1)², ±2(R + 1)², …` up to the largest not above the reach, and the last
> group is cut off at the reach: at `R = 1` a reach of 38 names 1, 2, 4, 8, 16 and 32, the last standing for 32
> through 38; at `R = 10` a reach of 250 names 1 through 10, 11 through 110 by elevens, 121 and 242, the last
> standing for 242 through 250. A reach of `r` holds
> ```
> offsets(r)  =  R · k  +  floor(r / (R + 1)^k)                 k = floor(log_(R + 1) r);  the distances from 1 to r
> ```
> so `offsets(r) + 1` offsets in time, zero and the past, and `2 · offsets(r) + 1` in any other dimension.

> **D5 — Adjacency.** An activation `b` is adjacent to an activation `a` when the following conditions hold:
>
> - `b` is in the same frame as `a` or in an earlier one.
> - `b` is within `a`'s reach in every activation dimension they share, time among them (D4).
> - `a` and `b` both stand on the apex (R27).
> - The neighborhood policy allows `b`'s kind beside `a`'s (D50).
>
> For example, in an image channel the pixels of one frame are adjacent to one another within reach. In space
> adjacency runs both ways. In time it runs only backward: what fires after `a` is never adjacent to it, and is
> held as a connection instead (D25).

Events and actions are compressed in the same patterns: above the base a pattern may name what was seen and
what was done together.

> **D50 — Neighborhood policy.** Declared once for the machine, as the alphabet is (D1). For it a neuron is of
> one of three kinds: an **event** or an **action**, a base neuron of that kind of dimension (D1), or an
> **interneuron**, whatever its body or its values hold (D2). The policy is two tables of booleans,
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
> **A coarse offset in a line may be held by more than one neighbor**, since it spans a range and several
> activations of one dimension can fall inside it (D6). A relation's offset is written rounded (D34), every
> neighbor whose exact offset from the relation's other node falls in it holds there, and `|p|` counts every
> node named. Above the base a `(dimension, position)` may itself carry several activations (D8), which this
> handles the same way and needs no rule of its own.
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
> the first could be for a neuron `s` in a stream reading `p a r i s`: at the base the reach is the radius, here
> 1 (D4), so `i` is its one neighbor and `p a r` are out of reach.
>
> **A neighborhood is the frame around the fired neuron**, since adjacency admits nothing later (D5), and every
> structural decision is made on it. The activation itself is not in it: its own instance is what a pattern
> covering it replaces, and is counted there (D22).
>
> **A neighbor that is a value neuron comes as its variable with its value** (D45): the neuron at its offset,
> and beside it the value that stood, a class's member or a parameter's member at each of its positions. The
> value is what a parameter compares (D44); the neuron is what a class admits, a function names, and a bid covers.

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

## 3.6 Apex

> **D49 — Apex.** The activations of a frame that no accepted bid covers, base neurons and interneurons alike:
> a base neuron no pattern found worth naming stands in it beside a child many patterns deep. It is the machine's whole state at
> that frame, and it has four names for the four things it is. It is the **frontier**, looking back: everything
> underneath it is recovered by expanding it, so it is what the file's body writes. It is the **situation**: what
> every neuron sees as its neighborhood (D5), what connects to what ran, and what votes. It is the
> **wavefront** of recognition: where the frame's levels stopped (R26), and from where the wave runs
> forward, each activation of it reading its own connections ahead, which it can do only because it is also
> the frontier, the neighborhood every neuron was built on. And it is the apex, the top of the hierarchy as it
> stands. The four are one set, and coverage takes a neuron out of it in every role at once.

> **D10 — Inhibition.** Coverage does three things to an activation, all from the frame it is covered in. It
> leaves the file: the body writes its coverer's call, not it. It is not heard: resolution reads its coverer's
> standing inferences, not its own (R46). And it stops connecting: from the frame after it is covered it writes
> nothing about what follows.
>
> **The last connection.** In the frame it is covered, the activation connects once more, to the neuron that
> covered it: the function's child, or the value neuron of a parameter that bid alone. The connection is at the
> offset from the activation's coordinate to its coverer's (D6), in time and in every other activation
> dimension, as every connection is (D25), and it takes that frame's reward (R31). Then the activation is
> silent. An activation standing at its coverer's own coordinate (D53) writes nothing, as one covered at age 0
> does. Without this connection the activation could never learn the neuron built over it:
> the child did not exist in the frames it was uncovered, and it is inhibited from the frame the child first
> fires. So a situation beside which a pattern has named what followed it learns that one neuron, the child, and
> from then on infers it; the child learns what follows.
>
> **What stands before coverage stands.** A neuron is covered only after it has fired (R27). Coverage acquired at
> a later age revokes nothing before it: the connections the activation strengthened while uncovered stand, and
> a reward for one of those frames still reaches them (R33). An activation covered already at age 0 writes
> nothing, ever: a connection is to what comes after (D25), and by then it is inhibited.
>
> **One exception, on being heard alone.** Every activation infers at its age 0, covered or not, and the machine
> keeps what it inferred (R45). A covered activation's inferences are heard only where its coverer has nothing
> for the frame ahead (R48), as a newly minted child has nothing until something has run under it. The covered
> activation still writes nothing and connects to nothing; only its coverer learns what ran. That is the whole
> of inhibition in the design.

## 3.7 File

> **D12 — File.** Two parts, both spanning the whole run. **The dictionary**: every neuron's table, its
> functions and variables, one line per pattern (D38). **The body**: every neuron no accepted bid
> covers (D10) — what each accepted bid fired among them, written as its call: its owner and which of its
> patterns, what each variable it carried held, and the neighbors it names that did not hold (error correction); and
> each bare neuron standing as itself. What a call fired stands on the frontier at what the call cost to write,
> and whatever covers it counts it at that and nothing more (D13).
>
> **Both types are in it.** An action the machine executed is a neuron that fired (D8), and it stands in the
> file exactly as an observed event does, compressed by the same patterns (D5).
>
> **It holds nothing about the future.** A neuron's connections — what followed its activations, event and
> action alike, and what the actions earned — are in no
> dictionary line and not in the body (D25), and neither is an inference that did not run (R32).
>
> **It holds nothing about the search either.** Populations, estimates and margins are the machine's and
> never the file's, because expanding an apex neuron needs the patterns and nothing else.
>
> **It holds the value.** Where a variable holds (D38), the body writes what stood there, a
> choice among the variable's values (D13), and expansion puts it back (R28). What a variable gives up is
> nothing the file holds: a function that names it has a child with one set of connections for every member, and
> the value neuron beside it holds the connections for that member (D45).

> **D13 — Prices.** Every cost in the design is part of a file, counted in **bits**. Each thing the file writes
> is a **field**, a choice among the options of its own alphabet, and a field costs `log₂` of its alphabet's
> size however often it is used: a fixed-length code, with one alphabet per kind of field.
> ```
> a base symbol             log₂ B                                   among the B base neurons the machine declares (D1):
>                                                                    every bucket of every event and action dimension
> a neuron, named in a line log₂ n                                   among the n neurons the machine holds
> an offset                 Σ over its dimensions  log₂ b            b the buckets of that dimension at the owner's reach r
>                                                                    (D6): offsets(r) + 1 in time and 2 · offsets(r) + 1
>                                                                    in any other
> which pattern             log₂ |patterns|                          which of the owner's patterns, of any kind
> which member              log₂ |K|                                 among the members of the variable K
> which neighbor failed     log₂ |p|                                 among the |p| nodes and positions the pattern names
> ```
> What each thing costs is the sum of the fields it writes:
> ```
> a function, in the table        =  a neuron per neighbor node + which pattern per variable node     written once
>                                 +  an offset per relation written, one fewer than its nodes (D34)
> a class, in the table           =  a neuron per member                                             written once
> a parameter, in the table       =  an offset per offset + a neuron per member                      written once
> a bid, on one occurrence        =  what its bidder cost + which pattern                            its owner, and which of its patterns
>                                 +  which member, per variable, per relation naming it
>                                 +  which neighbor failed, per node that did not hold
> ```
> A bid is a function with the variables it names, or a parameter alone (D31). The owner's activation is a node
> a function may name (D16), and it costs nothing in the line beyond the relation that names it, the neuron
> being the table's own. A function writes the relations that connect its nodes and no more, since the rest
> follow (D34): three nodes are two offsets, whichever two.
>
> **A symbol costs what it took to write.** That is the cost of an activation, and it is the same wherever the
> activation is counted:
> ```
> a base activation                       =  a base symbol
> a function's child                      =  what its bidder cost + which pattern + the neighbors that failed
> a value neuron beside a child           =  its variable's choice: which member
> the value neuron of a parameter alone   =  what its bidder cost + which pattern + its choice
> ```
> The machine keeps the cost with the activation, and a saved neighborhood keeps it with each neighbor (D18).
> **Covering an activation saves exactly what it cost.** So a child that only renames what it covers saves
> nothing: a class of two members covers a value neuron that cost one bit and writes one bit. And a class that
> admitted every base symbol would cover `log₂ B` and write `log₂ B`.
>
> **Each kind pays for what it says.** A function says which nodes and the offsets between them, once, and each
> occurrence says only that it happened: the owner, which of its patterns, and what each variable held. Where it
> stood is not said, since the layout is not priced. A class says which set, once, and leaves which member to
> each occurrence. A parameter says where, once, and leaves which member to each occurrence, paid once however
> many positions it stands at. A function that names a variable in two relations pays its choice twice.
>
> **A call pays its owner's name whether or not it covers the bidder.** Where the pattern names the owner's
> activation (D15), the bid covers the bidder, and what the bidder cost is on both sides and cancels (D22).
> Where it does not, the call still says whose pattern ran, and the owner's name is what that costs.
>
> **The layout is not priced.** Where a neuron stands is its place in the body, and how many things stand at one
> place or follow one instance is the layout's; no field says either. **Nothing about frequency enters**: a
> field costs the same however often its value is written.
>
> **`B` is declared and `n` is counted.** `B` is the size of the base alphabet and never moves (D1). `n` is the
> machine's count of its neurons, the base neurons and every neuron above them together: it starts at `B`, is
> raised when a neuron is created and lowered when one is deleted; it prices only the naming of a neuron in a table line, which points into the whole machine. The
> machine hands `n` to every neuron with its call (R20). Every other alphabet is the owner's own: its table and
> a variable's members.

> **D14 — File length.** Every neuron's table, and every activation on the frontier at what it cost to write
> (D13):
> ```
> L  =  Σ over every neuron's table:  its functions and variables              the dictionary
>    +  Σ over the activations on the frontier:  what each cost to write       the body
> ```
> There is one reading of it. A call is its owner, which of its patterns, what each variable it carried held and
> its failed neighbors, and what the call fired stands on the frontier at exactly that. A neuron above counts a
> child at what was written for it below, so nothing is saved by naming it again and nothing is gained by
> counting it twice. The election takes activations off the frontier at their cost and puts a call in their
> place at its price (R22), and a neuron's margin does the same over its own history (D30). Nothing computes
> `L`: every quantity the design uses is a **difference**, which is finite however long the run is.

## 3.8 Parameters

> **R4 — `H` and `R` are the only free parameters.** Two things are the designer's to pick: `H`, the depth of
> every neuron's history (D18), and `R`, one radius per activation dimension (D1), how far a base neuron sees in
> it. Nothing else is. The alphabet (D1) states the problem and is not a knob. The reach (D4),
> adjacency (D5) and the offsets (D6) are derived from `R`, not declared; the `n` every price reads is the
> machine's count of its neurons (D13). The neighborhood policy (D50) and the action policy (D51) are
> declarations like the alphabet: each states the kind of world the machine is in, and neither is a knob. **No
> rule introduces a constant, a threshold, a window or a cap of its own.**

The machine gives each activation dimension (D1) a **radius** `R`, a count of positions, `1` or more: how far a
base neuron sees in it (D4), and how every distance in it is rounded (D6). The radius is the resolution of the
layout, as the bucket count is the resolution of what is reported on it. For example, an image whose pixel is
to be told from the next has `R = 1` in `x` and `y`. A stream in which the fourth frame back matters as exactly
as the first has `R = 4` in time. Where this document writes a distance with no radius given, `R = 1`.

# 4. Neuron

What this section defines belongs to one neuron and is read nowhere else.

## 4.1 Interface

A neuron is a symbol, and a type (D2). It holds a patterns table, a
history of past and present observations, and its connections (D25). What it holds is defined in §4.2 through §4.7 and §4.13, and what it does with it in the
sections after.

**The machine reaches a neuron through six calls, and nothing else writes into one.** Each is specified where
it is used.

| Call                    | Description                                                                                                                                                       | References |
|-------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------|
| create neuron           | The machine creates it when an accepted bid needs one and none can be reused, holding nothing: a call neuron for a function, or a value neuron for the value a variable held. | R16, R43, R44 |
| process functions       | At its level: everything structural for the activations that fired this frame.                                                                             | §5, R20    |
| wire child to pattern   | Once the frame's last level has run, the machine points each function at the child it created or chose for it, and each variable's value at its value neuron.                                             | R16, R17, R44 |
| delete pattern neighbor | The machine removes a neuron that no longer exists from every pattern and saved activation that names it.                                                          | R38        |
| connect                 | After the frame's last level, reaching every open activation at age 1 and beyond: it delivers what stood on the apex this frame and any reward, and collects nothing.                                   | §7.1       |
| infer                   | After the frame's last level, reaching every activation at age 0: it delivers nothing and collects what the activation infers.                                                                        | §7.2       |

Every test the neuron runs is its own arithmetic over its own evidence, and it is never told what the board did
with its bids (R24).

Part II covers `process functions`: what a neuron does in the frame it fires.
Part IV covers the `connect` and `infer` calls, where a neuron learns what followed and infers what comes next.
Part V covers selection, where the machine chooses among those inferences one winner per dimension.

> **R1 — One decision point: the frame it fires.** A neuron is called once per level it fires at, for every
> activation of that level together, each at age 0, and everything structural happens in that call: it refreshes
> its history, recognizes the frame's neighborhoods, retires what no longer pays, builds every candidate that
> pays, merges what became identical, and returns the bids of each cover together with everything it retired
> (R20).

## 4.2 History

> **D18 — History.** A neuron's record of its own activations: the last `H` of them, oldest first, each at its
> coordinate, holding the neighborhood it observed in the frame it fired (D7), each neighbor at its exact
> offset (D6) and with what it cost to write, and the activation's own cost beside them, since a cover that
> names the owner is credited the activation too (D13, D20). It is the evidence every structural decision
> the neuron makes is read over, and it holds nothing else: the neuron's connections are not in the history and
> `H` does not bound them (R31).
>
> **`H` is declared once for the machine and is the same for every neuron.** It counts that neuron's **own
> activations**, not a stretch of run, so a neuron that fires constantly and one that fires rarely weigh their
> patterns against the same amount of evidence. **The ring is exactly `H` deep once filled.** A neuron may fire many
> times in one frame and not at all in the next (D8), so how many frames its `H` activations span is whatever its
> rate makes it, from part of one frame to a long stretch of the run. Nothing else anywhere is measured in frames.
>
> **Every activation the call hands it is recorded, covered or not** (R20). An arriving activation evicts the oldest,
> and only then.

## 4.3 Patterns

> **D32 — Table.** A neuron holds one **patterns table**, **the table** for short: its functions, classes and
> parameters (D38), one line per pattern, with each variable's members beside its line. Recognition reads it
> (D28); adding, retiring and merging write it (R15, R18, D43).

> **D16 — Node.** What a pattern names: a neuron (D2), a variable of the owner's table (D38), or the owner's own
> activation, which stands at offset zero in every neighborhood the owner records (D26).

> **D34 — Relation.** Two nodes (D16) and the offset between them (D6), one component per activation dimension
> the two share. In a line the offset is written rounded, as every written offset is, and the relation holds
> where the two nodes stand at any exact offset in the written offset's group. Relations compose: given the
> offset from `a` to `b` and from `b` to `c`, the offset from `a` to `c` follows, so a set of relations that
> connects its nodes fixes every other relation among them.

> **D15 — Pattern.** What a neuron **names**: a **function**, a set of relations (D34) that connects its nodes,
> one relation fewer than nodes, the rest following; a **parameter**, a set of offsets (D6) with no nodes, the
> same neuron assumed at both ends of each (D44); or a **class**, a set of nodes with no offsets (D41). A pattern
> fits inside the box a neighborhood spans (D5), and a function holds where one set of activations satisfies
> every relation it writes, anywhere in the owner's window. One that names the owner's activation holds at the
> owner and nowhere else, since the owner stands at offset zero, and one that does not holds at any place in the
> window. A pattern is the collapse of the uses it covers (D27), moves as they move (D29), and promotes a
> child: a function promotes one, a call neuron (R16), a variable one per value it has held, a value neuron
> (D45). Its `id` is its creation order within its owner's table, a handle that survives re-centering. It breaks
> one tie: where two of the owner's patterns measure the same in recognition (D28), the older `id` is taken. The
> election never reads it.
>
> **A pattern is not a neighborhood.** A neighborhood is what one activation saw where it fired, saved once and
> evicted `H` activations later; a pattern is what the neuron claims. A neighborhood is a fact, a pattern a claim.
> Neither is a frame: a frame is one column, either of these the whole window. What followed is in neither, and
> is held as the neuron's connections (D25).
>
> **A pattern has no place of its own.** Its relations are between the nodes it names, so the same arrangement
> seen one frame back and seen ten frames back is one pattern, and a neuron that looks back on one stretch of
> the run from several of its own activations sees one thing. Where it stood is where its child fires (D53),
> and is never written (D13).

**A pattern is one of two kinds: a function or a variable.** A variable is what changes from one occurrence to
the next, and it is of one of two types, each one set. A **class** is a set of members, and says that one of
them stands where a function names it, whichever. A **parameter** is a set of positions, and says that one and
the same neuron stands at all of them, whichever it is. A function is what stays fixed around them: its
relations, each two nodes and the offset between them, the nodes being neighbors, variables of its own table,
and the owner's activation where it names it. **Classes are a function's local variables and parameters are its
arguments**: a class is a place whose content is decided where the function runs, among what has stood there
before, and a parameter is one value handed in and used at every one of its positions. **An accepted bid is a
call**: the function's child fires over what the bid covered (D53), and one value neuron per variable fires
where the variable's value stood (§6.4). A parameter no function names is bid alone, and fires its value
neuron. A class has no place of its own, and stands only where a function names it.

> **D38 — Two kinds.** A pattern is a function or a variable. A function names relations (D34) over nodes
> (D16): neighbors, each a neuron (D26), variables of its owner's table, and the owner's activation where it
> names it. A variable leaves a neuron open, and is a class or a parameter:
>
> | Kind | Fixed | Open | Covers | Written once, in the table | Left open on each occurrence    |
> |------|-------|------|--------|----------------------------|---------------------------------|
> | a function | its relations: which nodes, and the offset between each pair it writes | what its variables hold | its neighbors, what stands where its variables hold, and the owner where it names it | a neuron or which pattern per node, and an offset per relation written | what each variable held         |
> | a class (D41) | its members | which member stands where a function names it | the neuron standing there | a neuron per member | which member, `log₂ \|K\|` |
> | a parameter (D44) | its offsets, between two or more positions | which neuron, the same at all of them | what stands at its positions | an offset per offset, and a neuron per member | which member, `log₂ \|K\|`      |
>
> A function says which nodes and the offsets between them, and which of its owner's variables stand among
> them. A class says which set the neuron is from, and nothing about where. A parameter says where, and that
> the neuron is the same at all of them, and nothing about which. **The two types are closed in opposite
> ways.** A class is closed on its members and open on where: a neuron that is not a member does not fit, and a
> function may name the class in any relation, and in several. A parameter is open on its members and closed on
> where: any neuron fits, seen before or not, and only where it stands at every position. A function owns none
> of the variables it names: every function of the table that names a variable names the same one, in one
> relation or in several, and it is paid for once, in its own line. A function may name no neighbor at all, and
> be nothing but variables brought together; over one variable and nothing else, not even the owner, it would
> write what the variable writes alone, and is never built (D30). A class stands in no cover but a function's
> (D31).
>
> **The owner's activation is a node like any other.** It stands at offset zero in every neighborhood (D26),
> and a function names it or does not. One that names it holds there alone and covers it: the bidder and what
> the bid names are then covered together. One that does not holds wherever its nodes stand in the window and
> covers them alone, and the owner stands as it was. On every occurrence a bid writes its owner and which of
> its patterns, what each variable it carries held, and which of the nodes it names did not hold (D13, D22). A
> variable writes what stood in it and passes it up, as a value neuron (D45). A function holds nothing but its
> relations: no branch, no loop. What branches is which pattern fires; what loops is the frame; what a variable
> held is above it, in its value neuron.

> **D41 — Class.** A variable that says **one of these stands here**: a set of neurons, its **members**, and
> nothing else. Where a function names it, as a node of a relation (D38), one member stands, and the function
> may name it in several relations, a member at each. Members stand in place of one another: two neurons that
> have stood together where the class is named are never both members. Its line is its members.
>
> **Its value** is the member that stood. Each member has a value neuron of its own (D45), and it fires where
> the member stood. Where no member stands where a relation names the class, that is a failed neighbor (D22),
> and the class does not hold there.
>
> **A class is not a neuron.** It is a pattern: found on its own uses (D33), named by the functions of its table
> wherever a member of it stands (D38), and deleted when its margin turns negative (R18) or it is left with one
> member (R41). What it saves is the choice among its members against what the neuron standing there cost to
> write (D13). Named in one relation by a function it is one choice there; named in two it is two, and a pair
> of members that recurs together is a function of those two neurons, found by the pair relation (D47), not a
> record the class keeps.
>
> **It grows and shrinks on one axis**, by its own collapse (D27). A neuron joins its members when what it
> covers, wherever the class is named, pays for its entry, and leaves when it no longer does; so what is
> learned about a member in one function holds in every other. It is closed on its members and open on where: a
> neuron that is not a member does not fit, and a function that has never named the class at an offset may.

> **D44 — Parameter.** A variable that says **the same one stands at all of these places**: a set of offsets
> (D6) between two or more **positions**, one offset fewer than positions, that hold one and the same neuron,
> whichever it is. The offsets carry no nodes, and compose as relations do (D34). Its line is its offsets, and
> beside them its **members**, the neurons that have stood at all of its positions within the history, each
> with its value neuron (D45). The list is a record and not a condition.
>
> **Its value** is the neuron that stood at its positions, **read through value neurons**: where a value neuron
> stands at a position, what stands there for the parameter is that variable's value (D45, D7), so one digit
> shown bare and the same digit held in a class are one and the same neuron to it. A position at which a
> variable's value neuron stands is recorded with that variable, so that expansion can put the value neuron
> back there (R28). Where the positions disagree, the value is the neuron at most of them, the latest on a tie,
> and every position that holds something else is a failed neighbor (D22).
>
> **A parameter is not a neuron.** It is a pattern: found where two offsets hold one neuron (D33), named by the
> functions of its table that hold where it does (D38), bid alone where none of them is in the cover (D31), and
> deleted when its margin turns negative (R18) or fewer than two positions remain (R41). It writes which member
> once, however many positions it has, so every position after the first is covered in full. For the neuron
> that stood it fires that member's value neuron, once, over its positions (D53, D45).
>
> **It is open on its members and closed on where.** Any neuron fits, seen before or not, and only where it
> stands at every position: a thing that comes for the first time is the parameter's value the first time, and
> is a member from then on (D27). An offset joins its positions when it comes to repeat the value, and leaves
> when it stops.
>
> **Bound from what has happened, placed at what has not.** When a function that names it is expanded with no
> value neuron beside it (R28), a parameter whose positions fall partly in frames that have happened and partly
> in frames to come takes its value from what stands at the first and places it at the second. That is how a
> value is carried forward in time, and it is the whole of copying.

## 4.4 Cover

What this section defines is shared: a neuron reads it over its own history and the machine over a frame.

A neuron explains each activation's neighborhood with patterns from its table. This section defines that
explanation: the cover, the owner of each neighbor, and the two readings of that, the coverage and the
residual. Every price in the design (D13) is counted off them.

> **D17 — Cover.** One per activation: the set of patterns chosen to explain its neighborhood. A pattern
> is a set of neighbors (D15), so it explains a subset of the neighbors of the neighborhood it names.

> **D19 — Owner.** One per neighbor of the activation: the pattern of the cover credited with it. A neighbor no
> pattern of the cover names has no owner. Over the activation the owners are a map keyed by its neighbors,
> each a neuron at its coordinate: it has the shape of the neighborhood, what fired, and beside each entry which
> pattern is paid for it (the owner). A neighbor that stands where a variable holds is owned by the function
> that names the variable, and by the parameter itself where it stands alone. The owner's own activation is in
> the map too, owned by the first pattern of the cover that names it (D38).

> **D20 — Coverage.** What one pattern of the cover is credited with: the neighbors it owns (D19), and the
> activation itself where the pattern names it and no pattern taken before it does (D28).
> ```
> coverage(p, O)   =   the neighbors of O owned by p, and the activation where p names it and no pattern was credited it first
> ```

> **D21 — Residual.** What no pattern of the cover is credited with: the neighbors with no owner (D19).
> ```
> residual(O)   =   the neighbors of O with no owner
> ```

## 4.5 Savings

> **D22 — Savings.** What one pattern saves over one activation: what it is credited with (D20), less what it
> costs, the fields its occurrence writes (D13). The pattern is a function with the variables it names, or a
> parameter alone.
> ```
> coverage(p, O)  =  what the neighbors of O that p covers cost, and the activation where p names it and nothing was credited it first
> price(p, O)     =  what its bidder cost + which pattern               its owner, and which of its patterns
>                 +  which member, per variable, at each offset named   what each variable it carries held
>                 +  failed · log₂ |p|                                  which of the nodes it names did not hold
> savings(p, O)    =  coverage(p, O)  −  price(p, O)
> ```
> `|p|` is the count of what the bid names: a function's nodes, a class counted once and a parameter at each
> of its positions per relation naming it, or a lone parameter's positions.
> What a pattern names holds or fails: a neighbor fails when its neuron is absent, a class where no member
> stands where a relation names it (D41), a parameter at each position where what stands differs from its
> value (D44), and the owner's activation never, since it is there wherever the owner reads. What a pattern
> fires on the apex stands in for what it covers; that is what it saves, and that is the coverage. What it costs
> is what its occurrence writes. What the bidder cost is on both sides and cancels where the pattern names the
> owner, for the first pattern taken; where it does not, it is on the price side alone (D13). A neuron the
> pattern does not name is not in the account at all: it costs what it cost whether the pattern exists or not.
>
> **A variable's part** is what it saves wherever it is in a cover, alone or inside a function, the call left out:
> ```
> part(x, O)      =  what the neighbors of O that x holds cost  −  its choice  −  failed · log₂ |x|
> ```
> Its choice is which member, once per relation naming it. `|x|` is a parameter's own positions, and one for a
> class: its part does not change with the function it stands in.
>
> **This is the only valuation in the design**, and it is read over three different sets, so the three numbers
> differ, and are meant to:
>
> | Read by                               | Over                  | Against                                  | Settles                            |
> |---------------------------------------|-----------------------|------------------------------------------|------------------------------------|
> | a neuron, in recognition (D28, D30)   | its patterns          | the residual of its history              | its covers, and so its bids        |
> | the machine, in the election (R22, R24) | this frame's bids   | the board                                | what is credited, and so the file  |
> | the machine, in the match (R49)       | one inferred pattern  | the frames its line reaches back into    | whether the inference is proposed  |

## 4.6 Greedy cover

One procedure appears twice in the design — once inside a neuron, over its own table, and once inside the
machine, over a frame's bids. It is stated here and cited from both.

> **D28 — Greedy cover.** The operation that covers a set of activations with **claimants**, each a set of
> named neurons read against one of those activations, and credits each covered neuron to exactly one claimant.
> It repeats:
>
> 1. **Measure** every claimant not yet taken against the still-uncovered neurons of its activation:
>    ```
>    coverage =  what the still-uncovered neurons of its activation it covers cost      the activation itself among them where the claimant names it
>    price    =  the fields its occurrence writes there (D22)
>    ```
> 2. **Take** the one with the highest `coverage / price` if `coverage > price`, and **stop** otherwise. It joins
>    its activation's cover (D17) and becomes the owner (D19) of the neurons it was measured on, which leave the
>    uncovered set; the round repeats from 1.
>
> **What it returns is the covers and the owners.** A cover is the claimants taken for that activation (D17);
> the owner of a covered neuron is the claimant of the round that took it (D19), so each is credited once; what
> no round took is the residual (D21). `price` is fixed by what fired and cannot change between rounds;
> `coverage` only falls.

**The two callers differ in what they cover and in how they break ties, in nothing else.** Each states its
tie-break where it runs (D15, R24).

| caller                     | claimants | to cover                      |
|----------------------------|---|-------------------------------|
| neuron - recognition (§5.2) | each function with its variables, then each parameter alone, against each activation | the residual of the history (D21) |
| machine - election (R24)  | bids | the board (§6.2) |

## 4.7 Bid

> **D31 — Bid.** What an activation offers the machine from its cover (D17), as the neuron holds it.
>
> **A function is bid with the variables it names, as one bid.** It cannot hold where they do not, so the
> machine takes it or leaves it whole. Functions of one cover that name the same variable depend on one another
> through it, and go as one bid, the variable carried once. **A parameter no function of the cover names is bid
> alone.** A class never is: it has no place but the offset a function names it at (D41).
>
> **A bid carries what its patterns name, and where each node stood.** It names the coordinate at which each
> node was found, read off the neighborhood with the owner at zero (D7), and what each variable held. The machine
> translates the coordinates by the bidder's own, credits the activations standing there (R22), and places the
> child over them (D53): nothing in the bid says where a child fires. An accepted bid covers what its patterns
> name: the bidder among them where a pattern names the owner (D38), credited once, to the first bid of that
> activation the election accepts (R24), and not otherwise.
>
> | Component   | Description                                                                      |
> |-------------|----------------------------------------------------------------------------------|
> | patterns    | the ids of the function or functions bid, or of the parameter alone; an id is its creation order (D15) |
> | relations   | the dictionary lines (D12): each function's relations, and the offsets of every parameter carried |
> | coordinates | per node that stood, where it stood, the owner at zero                           |
> | children    | per function, the id of the child it promotes; none until a bid of it has been accepted (R16) |
> | values      | per variable, per relation naming it, the member that stood, where it stood, and that value's value neuron, none until the machine has created it (D45) |

## 4.8 Relations

> **D47 — Seed relation.** Two nodes of the residual (D21) that bear on one another, counted over the history
> (D18) in **uses**: a use is a set of coordinates, and the same coordinates seen from two activations of the
> owner are one use (D27). The nodes are neighbors at their coordinates, variables holding there, and the
> owner's activation (D16). It is one of three kinds, and each is the evidence for one type of pattern:
> ```
> two nodes standing together                  a relation, a and b at δ from it (D34)                      a function of the two
> two offsets that hold one neuron             n at c, and n at c + δ, read through value neurons (D44)     a parameter of the offset δ
> one offset from a node where neurons vary    a at δ from c in one use, b at δ from c in another           a class of a and b, named at δ from c (D41)
> ```
> The first and the third hold between nodes of any kind, the owner's activation among them. The owner stands
> in every row at offset zero, so one neighbor that keeps standing at one offset from it is a function of the
> two, which at the base in a stream of text is what every row carries. The first is counted by the uses that
> hold the pair at that offset, wherever in the window the pair stands. The second is counted per offset,
> whichever neuron the two coordinates hold and wherever they stand. The third is across uses: `a` and `b` never
> stand together at `δ` from `c`, they stand there instead of each other, and it is counted by the residual
> neighbors holding either of them there, `a` and `b` being the two neurons that have stood there most and never
> in one use together. A coordinate a variable holds is read as that variable and as nothing else. The counts
> move as rows enter and leave the history and as neighbors are covered; how they are kept is the
> implementation's.

## 4.9 Collapse

> **D27 — Collapse.** The operation that grows a seed relation (D47) into a pattern, and re-decides what a
> pattern names once it exists. The seed sets the pattern's **uses**, the places where it holds, and everything
> it names is decided by its price over those uses (D13), from the uses alone. There is one collapse per type,
> and what a neighbor is worth is always what it cost to write (D13), read from the saved neighborhood.
>
> **A use is a place, not a row.** It is the set of coordinates the pattern's nodes stand at, and what stands
> around it is what every row of the history that saw those coordinates recorded there (D18). A stretch of the
> run that several activations of the owner looked back on is one use, however many rows hold it, and one row
> may hold a pattern at several places, each a use. The owner's activation of each row that saw a use is a node
> of that use, at its offset from the place.
>
> **A parameter.** A use is a set of positions, at the parameter's offsets from one another, where one neuron
> stands; where the positions disagree, its value there is the neuron at most of them, and each position that
> holds something else is a failed neighbor (D44). The seed is a pair of coordinates holding one neuron, and
> gives it that one offset. Its members are the neurons that have stood at all its positions within the
> history: **agreement is the whole test**, so a neuron never seen before is its value the first time it stands
> there, becomes a member, and pays its entry from then on. A member that no use of the history holds any longer
> leaves. Its positions are decided one by one, at the seed and at every call after; a seed position that fails
> leaves one, which is no parameter (R41). Every offset `δ` from a position already in is counted by the uses in
> which the coordinate there holds the parameter's value, and it is a position while what it covers there beats
> the failures where it disagrees and its place in the line:
> ```
> δ is a position when   what δ holds, over those uses   >   (uses − count(δ)) · log₂ |x|   +   an offset
> ```
> An offset that comes to repeat the value joins, and one that stops leaves. `|x|` counts the parameter's own
> positions (D22). A parameter left with fewer than two positions is deleted (R41).
>
> **A class.** A use is a coordinate where a function names the class, or, before any does, a coordinate at the
> seed's offset from its node, holding a member; where no member stands there, that is a failed neighbor (D41).
> The seed is an offset from a node where neurons vary, and gives the class the two neurons that have stood
> there most and never together. One test decides it, at the seed and at every call after; a seed member that
> fails leaves the seed one member, which is no class (R41), so the seed builds nothing and the pick names that
> neuron outright in its next stage (D33).
>
> *Its members.* A candidate is a neuron that has stood where the class is named and never in a use together
> with a member there: members stand in place of one another. It is counted wherever it stands at any offset the
> class is named at, and the candidates are taken in order of that count, each while what it covers beats its
> entry and the wider choice it gives every use the class already has:
> ```
> what m covers   >   a neuron   +   uses · log₂(|K′| / |K|)
> ```
> `K′` being the members with `m` added. Down the order the counts only fall and the widening only grows, so the
> first candidate that fails ends it. A member has to recur before it pays, and once it is a member it is one
> wherever the class is named. A member leaves by the same test read the other way. A class left with one member
> is an ordinary neighbor, and is deleted (R41).
>
> **A function.** The uses are the places whose residual holds the seed, each read in nodes (D33): a coordinate
> a variable holds is that variable, any other neighbor is the neuron standing there, and the owner's activation
> is itself. A candidate is a relation (D34) from a node already in the function to a node `x` of the residual
> at offset `δ` from it; the uses that count are those in which the coordinate at `δ` from that node is not
> named by another function of the cover (D19). Let `s` be their number and `count(x)` how many of them `x`
> holds in. Naming `x` saves what it covers in each of those, costs a failed neighbor in each of the other
> `s − count(x)`, and costs its entry in the line (D13):
> ```
> a neighbor covers    what it cost, in each use where it stands
> a variable covers    its part (D22), in each use where it holds, over all its positions at once
> the owner covers     what the owner's activation cost, in each use where it stands at δ
>
> x is taken when   what x covers, over those uses   >   (s − count(x)) · log₂ |p|   +   its entry in the line
> ```
> the entry being an offset for the relation, and a neuron for a neighbor node, which pattern for a variable
> node and nothing for the owner. Every relation is decided independently, from the input alone, so the result
> does not depend on the order they are decided in, and a node may enter by its relation to any node already in,
> since relations compose and the function is the same whichever is written (D34). A variable is named whole,
> and it fails at each of its positions where it does not hold. The owner's activation is decided like any node:
> a row that saw a use puts the owner at one offset from it, so it is counted at each offset rows saw the seed
> from, and named at the one where naming it pays, or at none. **A function chooses among the variables that
> exist and makes none**: variables are built by their own collapse, first (D33).
>
> **A coordinate a variable holds is that variable's.** Once a variable holds a coordinate, every function reads
> what stands there as the variable: it names the variable or leaves the coordinate alone, and it never names
> the neuron standing there. What stood there is not lost by that. It is in the variable's value neuron (D45). A
> neighbor another function of the cover names is skipped (D19).
>
> **A line never names a neuron built over its own child.** A neuron is built over a child when what it stands
> for passes through that child (D36): a pattern over the child, or over a neuron built over it. A function
> with a child skips such a neuron as a node, as it skips a neighbor another function names. A variable skips a
> neuron built over one of its own value neurons: it does not join a class's members, and a parameter does not
> take it as a value. So no neuron stands for something that contains itself, and every expansion reaches the
> base.

The collapse is the only operation that decides what a pattern names. It runs in two places:

- **Re-centering** (§4.10, D29) collapses the uses an existing pattern covers, at every call, for every kind.
- **Creating** (§5.4, D33) collapses a seed relation into a pattern. A candidate is in no cover, so every owned
  neighbor is skipped and it is built on the residual alone.

Skipping owned neighbors is what keeps patterns apart: a pattern grows only into the residual, never into what
another pattern owns, so two patterns covering the same activations cannot converge on one set of neighbors.

**Every relation is decided by the same uses.** A use says something at every offset from the seed, a neuron or
a silence, so the outermost relation is decided over the same `s` as the one beside the seed, less only the uses
skipped there. Each pattern's collapse has its own `s`; nothing is pooled across patterns, and no threshold,
smoothing or probability estimate enters.

## 4.10 Re-centering

> **D29 — Re-centering.** The operation that re-decides what a pattern names (D15) once the uses it covers
> have changed: a use lost when the last row that saw it is evicted from a full history (D18), or a new one
> admitted and covered with it (D28). It runs the collapse (D27) over the uses the pattern now covers (residual
> included), and updates the owners (D19) to what the pattern now holds. It never takes a neuron built over the
> pattern's own child (D27).

## 4.11 Margin

> **D30 — Margin.** What a pattern is worth over the history, less its dictionary line (D13). It is priced
> against what would cover its neighbors without it, never against nothing.
> ```
> benefit(f)  =  Σ over the uses f covers:                                   a function, with its variables
>                what the cover saves with f in it  −  what it saves without f                           D22
> benefit(x)  =  Σ over the uses x is in the cover of:  part(x, O)           a variable, alone or in a function
> cost(p)     =  its line                                                    D13
> margin(p)   =  benefit(p) − cost(p)
> ```
> **Without `f`**, the parameters it names bid alone where that pays (D31), its classes fall to the residual
> with the neurons standing in them, and so do its other neighbors. So a function is worth only what it adds: a
> residual neighbor in full, and for a parameter it names the call that parameter no longer makes. A function
> over one variable and nothing else, not even the owner, adds nothing. A variable is credited only the uses
> where it is part of a cover: a place where it merely could have held earns it nothing.
>
> It is the difference in the file (D14) between the file with the pattern and the file
> without it, and both terms are sums over what the neuron already holds.

A pattern is added only when its margin is strictly positive (R15) and retired only when strictly negative (R18).

## 4.12 Greedy pick

> **D33 — Greedy pick.** The operation that builds new patterns out of the residual of the history (D21), in
> two stages, one candidate at a time. In each, a relation (D47) is the seed and the collapse of its kind grows
> it (D27).
>
> 1. **Variables.** Of the two variable seeds, the offsets from a node where neurons vary and the pairs of
>    offsets that hold one neuron (D47), the one that would save most as it stands is taken first. What a seed
>    would save as it stands is the seed taken as if it were the whole pattern: D22's savings of it, what it
>    covers less what it writes, summed over the uses that hold it. Ties go to the higher count, then to the
>    older neuron, declaration order at the base and creation order above it, then to the shorter offset. The
>    collapse of its type grows it into a class or a parameter, its own test run on the seed's members or
>    positions before anything else (D27), and it is kept if what survives is a variable and its margin is
>    positive (D30); a seed whose members do not pay seeds nothing, and its neurons are named outright in the
>    next stage. A parameter already in the table with those offsets, or a class with those members, is used
>    rather than a new one. A variable that is kept **claims** the coordinates of its uses, and every seed from
>    then on counts only unclaimed coordinates, so no neighbor is evidence for two variables. Then the next
>    seed.
> 2. **Functions.** Each residual row is now read in **nodes** (D16): a coordinate a variable holds as that
>    variable, any other neighbor as the neuron there, and the owner's activation as itself. Of the relations of
>    two nodes standing together (D34), the one that would save most is taken first, read the same way as a
>    variable seed, with the same ties. The collapse grows it into a function over the uses that hold it, one
>    relation at a time, and it is priced by its margin (D30): over each use, what the cover saves with the
>    candidate in it less what it saves as it stands, where that is positive. A candidate that pays joins the
>    table and the covers it was priced on, owning what it names there (D19). Then the next relation.
>
> A seed counted fewer than twice seeds nothing, and the count is of uses, so one place in the run seen from
> several activations of the owner counts once (D27). Each stage ends when every relation of its kind has been
> tried or no residual is left. A parameter no function of a cover names stands in that cover alone, on the
> positions it holds, where its own bid pays (D28, D31). **What the pick returns is the patterns added.**

> **D43 — Merging.** Whenever a pattern is added or re-centered, it is compared with the table, exactly, and of
> each identical group the oldest is kept and the rest are retired (§5.5):
>
> - Two functions with the same nodes in the same relations are one function, their covered uses joined. One
>   that names the owner and one that does not are two functions. Relations compose (D34), so two lines that
>   write different relations over the same nodes and fix the same offsets among them are the same function.
> - Two classes with the same members are one class, and their value neurons for each member are one.
> - Two parameters with the same offsets are one parameter, and their value neurons for each member are one.

## 4.13 Connections

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
> exposures of every frame in its group. The neuron connected to may be a value
> neuron, and what an inference of it places is its value: a class's member where it stood, a parameter's at
> each of its positions (D45, R28).
> Nothing about any one activation is kept.
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

# 5. Process functions call

> **R20 — Call, in order.** Once per level it fires at, the machine asks one neuron for everything it owes
> that frame, handing it the count `n` of its neurons (D13) and **every activation of it that fired
> this frame with a neighborhood that is not empty**,
> each with its neighborhood, at age 0 (R1), every value neuron in it carrying its variable's values (D7),
> which the machine reads off the owner's table and the neuron could not. They are processed together. A neuron none of whose activations
> has a neighbor is not called; such an activation is still open (D9) and on the apex (R27) like any uncovered
> activation.
>
> | Step               | Description                                                                                                       |
> |--------------------|-------------------------------------------------------------------------------------------------------------------|
> | refresh history    | Evict as many of the oldest activations as the frame's need, then admit the frame's.                              |
> | recognize          | Cover the residual of the history with the table, each function with its variables and each parameter alone on what no function took, and re-center every pattern whose evidence changed. |
> | delete             | Retire every pattern whose margin is negative, every parameter left with fewer than two positions, and every class left with one member. |
> | create             | Build patterns: seed each on a relation, grow it by the collapse of its kind, price it; variables first, then functions over the rows as the variables rewrite them. |
> | merge              | Compare everything added or re-centered this call against its table, exactly; keep the oldest of each identical group and retire the rest. |
> | return             | The bids of each cover, a function with its variables and what they held, a parameter alone with what it held, each with where its nodes stood; and everything retired. |

The call runs before the election (R24).

## 5.1 Refresh history

The history holds `H` activations (D18). A frame bringing `A` activations evicts the `A` oldest once the history is
full, and fewer before, as many as it takes to make room; every pattern of an evicted cover loses it. The frame's
activations then join the history, whole (D7) and wholly residual (D21). A frame bringing more activations of one
neuron than `H` fills the history with the first `H` of them, in the order the input lays them out, and the rest
are not recorded; the machine reports it. `H` is the designer's to pick (R4), and a report says it was picked
below the most activations a neuron can have in one frame.

## 5.2 Recognize

Recognition is the procedure that chooses a cover for an activation's neighborhood (D17): the greedy cover
(D28) over the residual of the history (D21). It covers the whole residual, not the frame's activations alone:
every activation of the history whose residual changed since the last call is covered again, the frame's new
ones and those a retired or evicted pattern left neighbors uncovered in, and an activation nothing touched
keeps its cover, so the call's work is in what changed and not in `H`. The frame's own activations are always
among them, and their covers matter beyond the history: they are what the call returns as bids (§5.6), so what
recognition decides for them this call is what the election sees.

**A function fits where one set of activations satisfies every relation it writes, anywhere in the window, and
its variables hold; a class where a member stands where a relation names it; a parameter where one neuron
stands at all of its positions.** A parameter takes whatever neuron stands there, member or not, and one that
was not a member is one from that use on. A class takes members only (D27). Measuring a pattern against a
neighborhood, each node it names holds or fails (D22), and each that fails is a correction; where a failed node
would have stood follows from its relation to any node that holds. A function that does not name the owner may
fit at several places in one neighborhood, and each is a claimant.
Every function of the table is a claimant with the variables it names, and takes the activation itself where
it names it, with its neighbors (D20); a parameter is a claimant alone on the neighbors no function of the
cover took. What a variable held is what the bid carries (D31).

**Cold start is silence.** A pattern covering no activations names nothing, and a neuron with an
empty table covers nothing and bids nothing.

Every pattern whose covered activations changed, by eviction or by cover, re-centers (D29).

## 5.3 Delete

> **R18 — Retire.** After the frame's activations are recognized and the patterns re-centered (R20), retire
> every pattern whose margin (D30) is strictly negative:
> ```
> retire p  iff  margin(p) < 0
> ```
> Retiring is deletion from the table (D32). The pattern leaves that instant: it stops competing for a place in
> any cover, and the neurons it held fall to the residual (D21), where the next call's recognition may re-cover
> them (R20). A function that named a retired variable re-centers without it (D29). The retired patterns go on
> the return (§5.6), and the machine deletes their children (§6.5). The neuron keeps no retired state.

## 5.4 Create

The greedy pick (D33) runs over the residual of the history: each pattern is seeded on its relation, grown by the
collapse of its kind and priced by its margin, the variables first and then the functions over the rows as the
variables rewrite them.
Each that pays joins the table and the covers it was priced on. Every relation is tried.

> **R14 — Candidate.** A candidate `C` is what one round of the greedy pick builds (D33): the collapse of its
> kind over the uses whose residual holds the seed relation (D27). Nothing seeds it from outside, and
> nothing grows it a neighbor at a time. The seed is in every one of those uses and the relation has a
> count of at least two uses (D33), so **nothing is ever built on a single occurrence**, one place in the run
> seen from several activations of the owner being one use (D27), and the collapse takes the seed where it pays
> there. Everything else `C` names holds, in the residual, in enough of them to pay for itself:
> D27 skips what another function names as it does everywhere, so **a candidate is built on the residual**, and
> a function on the coordinates variables hold besides, which it names through the variable (D33, D27).
> The seed is what the table is failing on most, and the collapse settles every other node at once — the seed
> chooses the uses, and the uses decide every node.
>
> **Only what a pattern names is built, because that is all a pattern is.** What the child will be followed by
> is the child's own connections, formed by the child's own activations once it exists (D25). Nothing about it is
> decided here and nothing about it is priced.
>
> **The same history under the same covers yields the same `C`.** Covers are carried by the activations (D17),
> so the residual, and with it the seed, is a function of the ring and the covers it carries together.
>
> **What `C` is worth.** Against a use, `C` is worth what it adds to the cover that stands there (D30):
> ```
> savings(u)  =  what u's cover saves with C in it  −  what it saves as it stands           D22
>               and 0 where that is negative           C joins a cover only where it pays (D28)
> ```
> A use several rows hold is credited once, in the row whose activation is latest, since the file writes the
> place once (D14).
> With `C` in the cover, `C` owns the residual neighbors it names, and the variables it names stop bidding alone.
> **A candidate is never credited what another function names.** A
> neuron a function already names is not `C`'s to take — a candidate that fits a stretch of the run beautifully earns nothing
> for it if something already accounts for it.
>
> **The floor is the cover test and belongs to pricing only.** An activation whose cover `C` would not join contributes
> nothing to the benefit: a candidate has to be able to reach an activation before it pays in it.
>
> **What it is not.** It finds a local best and not the best `C` — choosing the pattern set is the
> facility-location problem and is not solvable exactly at any useful size. What it has instead is no free
> choice anywhere in it.

> **R15 — Add test.** The margin (D30) of a candidate, which is in no cover: it is credited what it adds to
> each activation's cover (R14), and only in the activations where it pays.
> ```
> benefit  =  Σ over the uses of the history:  savings(u)                              R14
> commit iff  benefit > C's entry in the table                                                             D13
> ```
> **An accepted candidate joins the cover of every use where its savings are positive**, as the owner of
> the residual it names there (D19), so its margin at birth is what the test counted.
>
> **An accepted add shortens the file**, against the table it was priced on.
>
> **The test is offline and complete.** Every activation in the ring has a whole neighborhood, so the question is
> asked over the same evidence the cover will use when `C` next competes for one. Nothing here is decided on half
> an activation and nothing later can hand `C` less than the test counted.
>
> **Every kind of pattern is kept by this test.**
>
> **The pick tries every relation** (D33). What a candidate that does not pay leaves uncovered is the next
> call's residual, and its relation is tried again then.

> **R16 — What a child is at birth.** The neuron proposes, the election decides, and the machine creates. A
> pattern is the neuron's own: it is added, used in covers and bid on the neuron's evidence alone. A function
> with no child is bid as it is, and the bid asks for one, a call neuron (D48), if it is accepted (D31). A variable
> has a child per value, a value neuron, and a bid carrying a value it has no child for asks for one the same
> way (R44). **The
> machine gives a child only to an accepted bid**, so no neuron exists for a pattern the board never bought, and
> it reuses the interneuron that already stands for what the child would (D36, R43) before it creates one. A
> new child has an id and no dimension (D2). It is created with
> **an empty table**: its own patterns are over what it will see, which it has not observed yet. Its *existence*
> is decided by the election, its *structure* by itself.
>
> **A neuron may hold many children, and they do not contend.** Each covers the part of an activation its
> pattern owns, and several of them may be promoted at one coordinate (D8). What they share is a parent and a
> coordinate, not a place on the board. **A child may have many parents**: every pattern wired to it promotes
> it, each from its own neuron's position, a function's child every function wired to it and a value neuron
> every variable (R43).
>
> **Release is the reverse**: the parent retires, the machine reclaims. The retired patterns go on the return
> (R20). A retired pattern that never had a child leaves nothing to reclaim, and one whose child other patterns
> still promote leaves the child alone (R38).
>
> **Creating and wiring are separate, and only wiring waits.** The machine creates or chooses the child the
> moment the bid is accepted — an id, its parent and an empty table — which is all the frame needs to activate
> it, at the coordinate the machine places its activation at (D53), and call it (R17). Once the frame's last level has run it points every pattern
> at the child created or chosen for it, in one pass over its own alphabet, and after the frame's connect and
> infer calls it reclaims every retired pattern's child now due (R38). Nothing in the frame reads that pointer
> except a later bid, so **the wait costs nothing**.

> **R17 — A pattern added in a call is bid in it.** The candidate joins the parent's table and the covers
> it was priced on in the call (R15), so every activation of the frame whose cover it joined bids it (R20), with
> no child, and the bid competes like any other. If it wins, the machine gives a function a child, reused or
> new (R43), and a variable a value neuron for what it held (R44), activates them at the next level and calls them
> there. A new child records its first neighborhood there and
> nothing more: its table is empty, and a candidate needs two neighborhoods (R14). If it loses, there is no child; the pattern stays in the
> table on its own margin (D30) and is bid again whenever it is in a cover.
>
> **It is born holding nothing.** No patterns, no history and no connections (R16). Everything it comes to
> hold is over the situations its parent's pattern actually took (D25), from its first activation on.

> **R41 — Life of a variable.** **Born** from a variable relation, a class from an offset from a node where
> neurons vary and a parameter from a pair of offsets that hold one neuron, when its collapse and its margin pay
> (D27, D33). **Changed** by its collapse, once per call: a class by its member test, a parameter by its position
> test, its members being whatever has stood at all its positions (D27). **Dead** when its margin turns negative
> (R18), and besides that a class when it is left with one member, which is an ordinary neighbor, and a
> parameter when fewer than two positions remain: it leaves the table.

> **R44 — Life of a value neuron.** **Given** when an accepted bid first carries the variable with that value,
> a function's bid or the variable's own: the bid carries the value with no child, as a function new to the board
> carries none, and the machine creates one (R16) and wires it to the variable's entry for that value (§4.1).
> **Fired** by the election and by nothing else (§6.4). **Dead** when the last variable wired to it is deleted or
> drops the value, its member leaving the parameter or the class (D27), or when the neuron it
> stands for is deleted: it
> goes on the death ledger and is deleted like any neuron nothing can fire again (R38).

## 5.5 Merge

Recognition re-centers, and creation adds, so either can leave two patterns in the table that are the same: two
functions with the same nodes in the same relations, two classes with the same members, two parameters with
the same offsets.
Merging compares everything added or re-centered this call against its own table, exactly (D43), and keeps
the oldest of each identical group, the lowest creation order, and with it the oldest children the group
holds, because those have accumulated histories, connections and estimates that a newer one has not.
A newer neuron with a history is never dropped for an older pattern with none. The rest are retired:

| identical | kept | retired |
|---|---|---|
| functions | the oldest, taking the others' covered uses, and, where it has no child, the child of the oldest of them that has one | the newer functions; a child no kept function points to dies when its last open activation's window closes (R38) |
| classes | the oldest, taking the others' place in the functions that named them, and, for each member both held, its value neuron | the newer classes and their value neurons for members the oldest also held; a member only a newer one held moves to the oldest with its value neuron |
| parameters | the oldest, taking the others' place in the functions that named them, and, for each member both held, its value neuron | the newer parameters and their value neurons for members the oldest also held; a member only a newer one held moves to the oldest with its value neuron |

What merging retires goes on the return with everything else retired this call (§5.6), so the machine deletes
the children on its ordinary pass.

## 5.6 Return

The call returns two lists: **the bids** of each of the frame's activations' covers (D31, R21), and **everything
retired** this call, by deletion (R18) or by merging (§5.5). A bid is lines of the table: a function's id, its
relations and its child, with each variable it names and what that variable held, or a parameter alone with
what it held, and where each node stood (D31). A function that has never won a bid has no child, and its bid says so: if it is accepted the machine
gives it a child, and every value the bid carries that has no value neuron one, reused or new (R16, R43, R44),
and wires them by the wire call (§4.1). A pattern retired is named by its id, and the machine deletes its child
or its value neurons, if it has any, when they are due (R18).

> **R21 — Bids of a cover.** An activation sends one bid (D31) per function of its cover, with the variables
> it names, functions that share a variable together in one; and one per parameter standing in the cover alone. A
> neuron covering nothing sends nothing.

# Part III — Past and present: the machine

# 6. Contraction

> **D23 — Contraction.** The machine covers the frontier with children, each taken when
> the neurons it covers cost more than its bid costs to state, never counting the dictionary line (R22, R24).
>
> **Covering everything is not the goal.** A neuron no accepted bid covers stays in the file as itself, at what it
> cost (D13), and that is the shorter file whenever no neuron could hold it for less. **What coverage varies is the
> file's length, never its fidelity.**
>
> It is **axis-general** — a pattern names neighbors at offsets, so a promoted neuron replaces a stretch of
> spacetime. Spatial contraction is the case where every offset is zero.

**The machine compresses a frame in this order and no other.** One level at a time: the first over what the
frame reported, each after it over what the level before bought, until a level fires nothing:

| Step              | Description                                                                                                   |
|-------------------|---------------------------------------------------------------------------------------------------------------|
| activate children | Fire every child and value neuron the level before bought, each placed over what its bid covered (D53); the uncovered stand on the apex. |
| process functions | Call every neuron with an activation at this level, value neurons included (§5).                              |
| elect bids        | Cover the frontier's activations that this level's bids name, by the greedy cover over the board.            |
| create children   | Give every function of an accepted bid that has no child one, and every value the bid carries that has no value neuron one: a child and the value neurons reused where an interneuron already stands for the same thing (D36, R43), created otherwise. Each fires at the next level. |

Then, once the last level has run:

| Step              | Description                                                                                                   |
|-------------------|---------------------------------------------------------------------------------------------------------------|
| wire children     | Point every child created this frame at its pattern, and every variable's value at its value neuron (R16).    |
| delete children   | After the frame's connect and infer calls (§7), delete every child that is due (R38).                        |

## 6.1 Process levels

**A level is explained as a set of function calls with their arguments.** Every accepted bid is a call: a
function's child, and beside it one value neuron per variable the function names, with what the variable held
(D45); or, for a parameter bid alone, its value neuron. What stands above is made of the calls and their
arguments (§6.4): a function there over a child alone holds for every value the call was given, and one over the
child and a value neuron holds for that one value.

> **R26 — Levels.** A frame is compressed one level at a time. The first level is every base activation the
> frame reported, all of them together: they run `process functions` and offer, and the election settles which
> bids are bought. The next level is the children those bids bought: they fire, run `process functions` and
> offer, and so on upward. **When a level fires nothing, the frame's levels end there.** Nothing declares the
> depth and nothing caps it. A level belongs to a frame and not to a neuron: a neuron fires at whatever level
> buys its bid, and one neuron may fire at different levels of different frames.
>
> **Within a level the order is activate, call, elect, create**, and it cannot be otherwise: what fires is what
> the level before bought, the bids are what the election is over, and a child is created only for a bid the
> election accepted (R16). Nothing in that order leaves the level or the frame.
>
> Every neuron runs the same rule at its own reach (D4). **Compression is spatio-temporal for every neuron, in
> one pass**: a pattern may name neighbors in its own frame, in earlier ones, beside it in space, or in a mix.

## 6.2 Election

> **R22 — What a bid covers, and what it costs.** The neuron sends the bid, its patterns and what each variable
> held (D31), and nothing else. The machine holds the frame, so it reads that one object against what fired
> and derives both numbers.
> ```
> the bid   its patterns, each function's child or none, what each variable it carries held, and where each
>           node stood, the owner at zero                                                               (D31)
> covered   the neurons it covers that fired and no earlier bid covers, its bidder among them where a pattern
>           names the owner — each at what it cost
> price     the fields its call writes (D13): what its bidder cost, which pattern per function it carries,
>           what each variable held, and which of the neurons it names in those same frames did not fire
> ```
> The bid's coordinates are translated by the bidder's own, so the machine finds each node where the bid says it
> stood, and the child it places is placed over them (D53), by a rule the neuron never reads.
> `coverage − price` is the savings over stating what it covers flat, D22's expression over the machine's population.
>
> **A neuron that fired and the bid does not name belongs to neither side.** It stands in the file as itself if
> nothing covers it (D23), and the same way if this child is promoted, since the child does not name it — its own
> cost either way, so it cancels before the test begins, and charging it here would count it twice against the
> uncovered term of the same sum (§6.2).
>
> **Coverage changes the credit and never the price.** A neuron the bid names that fired and another bid
> already covers is credited to no one and charged nothing: it fired, so it was never among the neurons named
> and absent. A neuron the bid names that did not fire is charged a failed neighbor whether or not another neuron is
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

> **D24 — Coverage set.** The machine keeps which accepted bid was credited each covered
> activation:
> ```
> coverage set    which accepted bid holds each subsumed active activation
>                 the owners             one owner per activation; a settled activation never changes owner
> ```
> **An activation is named by its full coordinate**, dimension and position together, so two activations of one
> neuron at two positions never contend. The coverage set spans one frame more than the longest reach in time
> among the neurons the machine holds — no bid reaches further back — and the box every other activation
> dimension gives, and ages out with it, and over the same span the machine keeps the action that ran at each
> frame in each action channel, for the shares a reward sends back (R33). **The machine holds nothing on the
> scale of the run.**
>
> Ownership is about **credit**: a neuron is a fact that needs paying for exactly once, so it is settled
> once and never revisited (R23). It is not about naming: a neuron expands to everything its pattern names,
> credited or not (R24).

> **R23 — This frame's bids against the board as it stands.** Only neurons no earlier frame's election has
> credited are in play, so an activation already paid for is not paid for twice. **No earlier promotion is ever
> re-scored.** Within the electing frame an activation is credited once, to the first accepted bid that names it
> (R24), and never moves.
>
> **Earlier bidders have priority.** A bid at `f` wins a neuron at `f − 2` before a better bid at `f + 1` can
> name it, because the earlier election settled it and nothing re-elects the past. That is the price of never
> revisiting a frame, and the design pays it.

**The file over one frame is the calls promoted plus the neurons no bid covered**: the price of every accepted bid,
and what each neuron no bid covered cost, the body half of `L` (D14) over the frames the election can see.
**The two terms do not overlap**: a neuron a promoted neuron fails to name is in the second and not the first,
which is why the price counts only the neighbors named and absent (R22). The dictionary half is D30's, and neither
test touches the other's sum.

**That sum is the objective; R24 is the procedure that serves it.** **Nothing anywhere forms a subset of bids
and scores it** — bids are taken one at a time, each measured against what the ones before it left, and the
election stops at the first that does not pay. **The election does not minimize the sum**: it is greedy, and
returns a good solution, not a proved minimum. **Every neuron a bid covers ends up credited to exactly one
bid**, which is what stops an activation being paid for twice.

> **R24 — The election is D28, run by the machine.** The claimants are this frame's bids and what they cover is
> the **free set**: every activation on the frontier (R27), at its own full coordinate — frame and position — that
> some bid names and no earlier election has credited (R23).
>
> Bids arrive naming where each node stood, the owner at zero (D31), so each is resolved on the board by the
> bidder's coordinate before the first round.
> `price` is R22's. **Two bids that measure the same are told apart in this order**: a bid carrying a child
> before one carrying none; between two with children, the older child; between two with none, the older
> bidder; and between two activations of one neuron, the earlier coordinate, the earlier frame first and then
> the smaller position, the dimensions in the order the machine declares them (D1). Older is the smaller id
> (D2). Then D28 runs, and what it takes are the accepted bids.
>
> **The bound is structural.** A bid's price includes what its bidder cost, so a bid is taken only when it
> covers more than that: at least one neighbor beside the bidder where its pattern names the owner, and more
> than the owner's name where it does not (D13). Each round takes one bid, so the rounds are never more than
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
> **Outcome**: accepted bids are promoted, each **whole** — its functions' children and its variables' value
> neurons together, and a child expands to everything its function names, credited or not — the neurons credited to
> them are subsumed, and every active neuron no accepted bid covers stands as itself. **The election delivers
> nothing to anyone**: it writes the coverage set and stops. No neuron is told which of its bids were bought, what
> they were credited, or what they lost; a neuron's history is what it saw, and the board is the machine's.

## 6.3 Create children

Every function of an accepted bid that has no child is given one, and every value the bid carries that has no
value neuron is given one. A bid that lost is given nothing.

> **R43 — Reuse or new.** The candidate for reuse is the interneuron that stands for what the accepted bid's
> child would, found through the index (D36), wherever it is: a neuron that bid this frame, one that fired
> frames ago and has stood for the same thing since, or one that has not fired within any open window. A stretch
> that comes round again therefore fires the neuron minted when it first came round, and not a new one above it.
>
> **Identical bids share one child.** Two bids are identical when their children would stand for one thing: each
> names exactly the activations the other names, the bidder among them where a pattern names the owner, at the
> same price (D31), two lines stating one set of activations from two of its neurons, which the machine places
> at one coordinate (D53). Every neuron the accepted bid covered was called this frame and bid every line of its cover
> (R21), so such a line is among this election's bids, and a pattern that does not name its owner is the common
> case, since every neuron whose window held the place saw the same nodes there. There are three cases, and
> each ends with one child.
>
> - One carries a child and the other none: the one with none is wired to that child.
> - Neither carries a child: the first accepted is given one (R16), and the other is wired to it.
> - Both carry children: the younger child's patterns are rewired to the older child, every one of them, and
>   the younger, with no pattern wired to it, dies when its last open activation's window closes (R38).
>
> A bid that measured better than the accepted one would have been taken before it (D28), so reuse among the
> board's bids is only ever a tie; a bid that does better than every bid carrying a child states a different set
> of activations, and gets a child of its own unless some neuron already stands for it.
>
> **A variable's children are reused with the function's.** The two bids name the same ground, so the machine
> also has what each variable of either covered on the board. Where a variable of the accepted bid covered the
> same activations as a variable of the other bid, and is of the same kind, it is the same variable seen from
> another owner: the accepted bid's variable is wired to the other variable's value neurons, as one set, and
> every value either passes from then on has one value neuron, created by whichever passes it first (D45). Each
> variable stays in its own table, with its own members or positions. A variable with no counterpart in the
> other bid keeps value neurons of its own, and a class and a parameter are never one. A value neuron is reused
> whenever a variable wired to it passes that value again.
>
> **Otherwise the machine creates one**: an id, its parent and an empty table (R16), its first activation
> placed as every activation of it is (D53); a value neuron the same way, with no dimension (D45, R44).
>
> **Nothing is sent back.** The pattern stays the neuron's own line and keeps re-centering on its own history
> (D29); only the pointer is shared. Lines wired to one child may drift apart, as one line drifts over time,
> and the child stands for all of them. The neuron optimizes its history and the machine its window.

Once the frame's last level has run, the machine points every function at the child created or chosen for it, and every
variable's value at its value neuron, by the wire call (§4.1).

## 6.4 Activate children

A child fires at the level after the one its bid was accepted at, before that level is called. What an accepted bid carries
fires then where the machine places it (D53): the child of each function at the latest frame the bid covered
and the center of the box its nodes span in every other dimension, and one value neuron per variable where its
value stood (D45). A parameter bid alone fires its value neuron and nothing else. A pattern that does not name
its owner may put its child frames back from the frame being compressed, where its nodes stood; the child is in
this frame's levels all the same, and stands on the apex of the frame of its coordinate from now on (R27).
Nothing sees a child until it fires (D5). A child's activation records
the function whose bid fired it, and a value neuron's the variable, since either may have several (R43), and
each is expanded through that one (R28). Every activation no accepted bid covers stands as itself.

> **R27 — The apex is a frontier.** It is every active neuron **no accepted bid covers** (D49) — the
> uncovered set, base neurons and interneurons alike — so a base neuron no pattern found worth naming stands in it beside a
> child many patterns deep. This is the frontier the file's body writes, **the one every neuron sees as its neighborhood**
> (D5), **the one that learns what ran** (D25), and **the one that votes** (§8.4): the uncovered set does all four,
> and coverage silences a neuron in every one of them at once, its one last connection aside (D10). A reward for a
> frame already written reaches its connection regardless (R33). Everything underneath the current frontier is
> recovered by expanding it.
>
> **Uncovered, not childless.** A neuron that offered a child and had its bid declined with nothing else
> covering it is still on the frontier, and stands in the file as its own line.
>
> **Coverage is acquired late and never revoked.** An activation firing at `g` is uncovered until some bid
> takes it, and a bid can take it for as long as it lies within the bidder's reach (D4), so an activation may
> speak for a few frames and then fall silent. **No accepted bid is ever dropped** (R23).

The frontier cuts across the hierarchy:

```
                                    ┌────── ▣ ──────┐
                          ┌─── ▣ ───┐               │
                ┌─ ▣ ─┐   │         │       │       │
                a     b   c         d   e   f   g   h        i     j
                                                             ▣     ▣

   frontier  =  { the child over (a,b),  the child over (c,d),  the child over (e,f,g,h),  i,  j }
```

**Events and actions run together within a level.** An action fires in the same column as the events it runs
alongside (D8), so it is recognized and covered by the rule they are, in the same patterns (D5). **The
connection is not formed here**: it names what stands on the apex, which is known only once the frame's last level has
settled, so it is recorded in the `connect` call instead (R31).

---

## 6.5 Delete children

> **R38 — A child dies when its last parent has retired and the last window over it closes.** A pattern the
> neuron retired (R18) names a child the machine still holds, unless no bid of it was ever accepted and there
> is nothing to delete (R16). The machine wired every pattern that promotes the child (R43), so it knows when
> the retired one was the last; while another remains the child lives and only the retired pattern goes. The
> neuron says nothing about when the child should go: it sees its own open activations and not the children
> promoted off them, while the machine sees both.
> ```
> death frame   =   when the last open activation of the child, or of any neuron in its subtree, closes its window
>               =   this frame, when none is open
> ```
> That set only shrinks. Once every pattern wired to the child has retired nothing can fire it again, and nothing built afterward can name it either, because a pattern is built out of
> what is firing (R26). An activation's window closes at its reach in time (D9), so the wait is at most the
> longest reach in time among the child and the neurons built on it.
>
> **Deleting is after the frame's connect and infer calls** (§7), so the last exposure an activation writes at
> age `reach_t` is taken before its neuron goes: every frame, after those calls, the machine reads the **death
> ledger** and takes everything due — the pattern, its child neuron and that neuron's **subtree** together, at
> once, and what named them scrubbed with them by the delete call (§4.1). The subtree is every pattern in the
> child's table, every child of those patterns that has no parent outside the subtree, and so on down; a child
> with a parent outside it stays, and only the pattern wired to it from inside goes. A pattern retired this frame with nothing open dies on
> this frame's pass; one whose child is still open waits exactly as long as what is built above it needs, and not
> a frame longer. **Nothing traces who is naming what**: the machine settles the question off the board it
> already keeps.
>
> **The ledger holds the pattern, not a handle to it.** A child is stated by the patterns wired to it (D12),
> and each activation of it is expanded through the pattern whose bid fired it (R28, §6.4). Until the
> windows of the activations a retired pattern fired have closed, neurons above them still cover them and the apex may still
> expand them, so that pattern has to stay readable after the table stops covering with it, whether or not the
> child lives on.

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

**A child fired at a coordinate frames back is called for the frames it was not there for.** Its pattern did
not name its owner, and its nodes stood before the frame being compressed (D53), so it opens at the age its
coordinate gives it (D9). In the frame it fires it is both infer and connect at once: the machine takes its
inferences, as from any activation in the frame it fires, and then hands it, for each frame from the one after
its coordinate to this one, that frame's apex within its reach and that frame's rewards at their distances,
read from the coverage set and the actions the machine keeps (D24), as if it had been open then. From the next
frame on it is called as any open activation is.

The neuron chooses nothing in either call and is told nothing of what came of its inferences. Choosing among the
inferences of every activation is the machine's, and is Part V (§8).

> **R29 — Two frames: infer, then execute and reward.** What is chosen in one frame is output for the next, and
> what ran, with what it earned, arrives with that frame.
> ```
> f      infer     the frame's events are recognized, `infer` returns the inferences,
>                  the inference resolves (R47), and an action is output for the frame ahead
> f + 1  execute   the report names what ran, and its neuron fires strong in this frame's column
>        reward    alongside this frame's events — every uncovered activation open connects to the
>                  apex; what the action earned arrives as this frame's input, and the connection
>                  it strengthens takes the reward in the same write (R31)
> ```
> **The reward is part of the frame the action ran in.** The environment reports what it observed and what the
> action in effect during that frame earned together, so an action is never on the books without its outcome,
> and no activation has to stay open for a reward that arrives later than the action it pays for (D9).

## 7.1 Connection

**The machine calls `connect` on every open activation at age 1 and beyond**, once per frame of its window, with
two things:

```
the apex          every activation standing on the apex this frame within the
                   activation's reach in every dimension but time (R27, D25). **Only if the
                   activation is uncovered at this frame.** One an accepted bid covered this
                   frame is handed the neuron that covered it and nothing else; one covered
                   earlier writes nothing more                                            D10, D25, R31
the reward         any reward that arrived, for the actions of this frame and for any earlier
                   frame the reward spans, at the distance each one names, with the action
                   that ran at that frame in each channel it pays (D24)                        R33
```

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
> **Every neuron holds them, base neurons included.** No operation derives what a child was worth from what its
> parent earned or the reverse, so a connection held only at the frontier would be lost at the next mint. Every
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
> `S` frames,
> ```
> share(d)  =  reward · (S − d) / S            d = 0 … S − 1
> ```
> so the frame the reward arrived in takes the whole reward, the far end of the span takes `reward / S`, and
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
> how recent its exposures are — a child's connections are over the one situation its parent's pattern names (R35),
> and a changed situation is answered by a new child, never by forgetting.

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
> other activation dimension from its position, as a neighbor is placed (R28). Expanded through its dictionary
> line it places base symbols at the frames back from there (R30), every one carrying the strength and the
> estimate of the connection it came from. A connection to a call neuron places a call (D37), weak, its
> parameters bound from what stands at those of their positions that have happened (D44). What its expansion
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
| match    | Read each standing inference's line against the frames it reaches back into; propose it where the line would pay there, strike it where it would not (R49). |
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
> what its bid covered: the nodes its pattern names, the one whose bid fired the activation (§6.4), placed
> around the activation's coordinate as the child was placed over them (D53), the latest node at the
> activation's frame and the nodes centered on its position in every other dimension, each other node by its
> relations (D34), the owner's activation among them where the pattern names it — offsets compose because each
> is a difference of activation coordinates (D2) — repeated down to base symbols. A pattern that does not name
> its owner puts nothing back for the owner: it never said where the owner stood.
> ```
> A placed at f:      A's line, in v's table, is {u at −1 from v}              → v at f,   u at f−1
> P placed at f+4:    P's line, in b's table, is {C at −2 from b},
>                     C's line, in c's table, is {d at −1 from c}              → b at f+4, c at f+2, d at f+1
> Q placed at f+4:    Q's line, in b's table, is {u at −3 from C}              → c at f+4, d at f+3, u at f+1, and no b
> ```
> **A coarse offset expands to its rounded coordinate.** A neighbor named within the radius is placed where it
> fired. One named beyond the radius is placed at exactly the distance its offset names, whatever distance in
> the group it fired at, and several neighbors at one coarse offset (D6) are each placed there. **The rounding composes.** Each expansion down adds its own group's slack, so a neuron
> places the base symbols of its farthest neighbors to within the sum of the groups along the path: the longer
> its span, the coarser its far placements. That is the loss the file carries (§1), and expansion reports it
> faithfully rather than hiding it.
>
> **This is the one expansion in the design.** It recovers the run from the file (D12): a parameter expands to
> the member its value neuron names, at every one of its positions, and a class to the member its value neuron
> names, where the function named it. And it turns an inferred
> pattern into a program (R30): the actions it places run, and the events it places fire weakly beside them
> (D40). It reads dictionary lines only. Recognition runs it backward: a pattern fit by what
> fired says that this function ran here, with these arguments, and the neuron above writes the call and its
> values. **What
> travels down with a symbol is
> what the connection carried**: every base symbol an inference's expansion places carries the strength and the
> estimate of the connection it came from, and nothing is re-weighted on the way down.
>
> **Expansion substitutes the arguments.** A function's child expands to its nodes: its neighbors, the owner
> where it names it, and what its variables held, read from the value neurons its bid fired: a parameter to its
> member at every one of its positions, a class to its member at each offset the function named it at. Where an
> inferred call brought no value neuron,
> a parameter is bound from what stands at those of its positions that have happened, read through value
> neurons (D44), and placed at the rest: that is how a value is carried forward. At a position the parameter
> reads through a variable, what is placed is that variable's value neuron for the member (D45); elsewhere, the
> member itself. Where none of its positions has happened it places its most frequent
> member, and a class with no value neuron places its most frequent member, the oldest on a tie. A value
> neuron standing alone expands to its value at every position of the parameter whose bid fired it. A function
> above that holds a child alone expands to the call with its arguments left to be bound, and one that holds
> a value neuron beside the child expands to the call with that value. A neighbor that is itself an interneuron is
> expanded the same way from where it is placed. Recognition is this substitution run backward: a parameter is fit
> by `t` standing at every one of its positions, and the function above receives its value neuron for `t`, beside the
> child of the function that named it (D45). **Two words, never one.** A pattern is **expanded**, downward, into
> what it covers; a connection is **inferred**, forward, into the frames ahead (D25); what a call does and what
> follows it are different directions.
>
> **A connection is placed the way a neighbor is.** A connection at offset `b`, read by an activation at its age
> 0 (R36), puts the neuron it names `b` frames after the activation's frame — that is where it completes — and
> its expansion hangs from there, its base symbols at the frames back from it. At a later age `a`, the raw count
> of frames elapsed (D9), that completion is `b − a` frames ahead: `b` is the connection's offset, a D6-rounded
> number, and `a` is never rounded. **A coarse
> offset is not a window.** Its action completes at
> `b`, not somewhere in the group `b` stands for, exactly as a neighbor named at `−b` is placed at `−b` and nowhere
> else. So the base symbols of a long program reach the frame ahead one at a time, in order, each at exactly one age,
> from one connection and with nothing held: what an activation places beyond the frame ahead it places again
> next frame, one frame nearer.

> **R30 — Execution is an expansion**, of the selected pattern (R47) through its dictionary line (R28). A high
> pattern becomes its members at the offsets its line records, down to base symbols. Its base actions are output,
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
> like a bought one: it runs `process functions`, may be covered (§6.1), and every uncovered activation open
> at that frame connects to it (D25). While its
> program runs, what stands on the apex at each frame is the base action it placed there, or a lower pattern as
> it completes; the whole stands on the apex only in the frame it completes, and what the program earned reaches it through the span a
> reward names (R33).

## 8.3 Match

> **R49 — Match.** The machine reads one standing inference (R45) against the frames its line reaches back
> into, and proposes it where the line would pay there and strikes it where it would not. It runs every frame
> the inference stands, over the part of the line that has come to lie in the past. Nothing competes in it,
> nothing is credited and nothing is written: it is the one valuation (D22) read over its third set (§4.5).
>
> **What is read.** The inferred child's function, as its dictionary line stands (R28): its neighbors, each at
> an offset, and its variables, each at its positions, placed back from the completion the inference names, one
> expansion down and no deeper. The bidder, and whatever is placed at the frame ahead or beyond, is what the
> inference proposes and is not judged. Against them stands the apex of each frame within the longest reach in
> time, as the machine keeps it (D24).
>
> **What each placed node is worth.**
> ```
> a neighbor standing at its offset                     credits what it cost
> a neighbor standing within reach at another offset    credits what it cost, and charges an offset (D13)
> a neighbor standing nowhere within reach              charges a failed neighbor, log₂ |p|
> a variable                                            bound from what stands at its positions that have happened (D44),
>                                                       charging its choice; a position holding nothing it could take
>                                                       charges a failed neighbor
> ```
> An offset is read as D6 reads it: exact within the radius, and beyond it as its whole group, composed down the
> expansion as the placement is, so a neighbor named at a coarse offset stands at its offset anywhere in the
> group. A neighbor found at a different distance is a correction that says where, and it costs what an offset
> costs to write, near or far: what makes a near miss cheap is that within the group it costs nothing.
>
> **The test.** The inference is proposed when what the placed nodes credit exceeds which pattern and what
> they charge, and struck otherwise. That is D22's test with the bidder on neither side, since the latest node,
> and the owner's activation where the line names it, is ahead.
>
> **What it decides.** Proposed, the inference's placements ahead contend for their positions (R46, R47).
> Struck, it leaves the standing inferences (R45). An inference that was proposed and stops matching as its
> placements come due is struck then, which is the whole of how a plan ends before its span does.
>
> **Where it stands among the three.** Recognition reads a pattern against the neighborhood its owner saw and
> chooses covers (§5.2). The election reads bids against the board and credits them (R24). The match reads one
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
> neuron's estimate is a marginal over every situation it has stood in; a child's is over those its parent's
> pattern recurs in, one situation; so the same frame reads differently to a base neuron, to the child covering
> part of it and to the child covering the whole. **A minted pattern is how one recurring situation acquires an estimate of its own**
> — which is why a covered neuron is silenced (D10) and its coverer speaks instead, and the only reason a base
> neuron's marginal ever speaks is that nothing more specific was bought over it.
>
> **Every apex activation votes**, base actions included. A pattern that names what was done beside what was
> seen votes knowing the situation; a child over actions alone, or a base action, votes what followed it, a habit,
> and reward prices the habit like anything else. What an inference places in an event dimension is resolved
> the same way, and the winner is output (R47, R30).
>
> **Nothing runs where nothing infers.** An action dimension no inference reaches is output nothing: no call
> runs, nothing fires in it, and the frame carries silence there as an event dimension with nothing to report
> does. The first action in a dimension is the world's. An action the environment executes on its own is a
> lesson: it stands on the apex of its frame, every uncovered activation connects to it with its reward by
> the ordinary path (R31), and from then on it is an action the machine can infer. A neuron is born holding no
> connection at all — a base neuron at cold start and a freshly
> minted pattern alike say nothing about the next action until something has run under them, and what they
> then hold is whatever was executed. R37's walk is the only thing that wires an action ahead of its running,
> one at a time and only where the best known has been judged and found wanting.

> **R46 — Electorate.** At frame `f` the voters on the frame ahead, `f + 1` (R35), are the open activations
> the machine holds at `f` (D9), less every activation an accepted bid covers (D10), plus every covered
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
> in theirs, down to the base; nothing is output where nothing down to the base has anything. A newly minted
> child therefore starts with its parents' voices: they say what usually followed them, it learns what ran under
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
> situation name it in a pattern (§5.4), and that pattern's child is a new interneuron with the walked action
> inside it, found as a lesson's action is found, one base action at a time and then named. What a call's variables hold is never searched, it is
> whatever stands at their positions when the call runs (D41, D44).
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
| Alternation | [algorithm-xy.md](algorithm-xy.md) | A stream at radius 4. A neuron that keeps seeing a different pair of letters holds one parameter, the same letter two frames apart, and one function that names it twice, learned from the pairs that go by once: its child is the alternation, one neuron for every such pair, and the value neurons behind it say which pair. A pair that comes often is named by its own letters' patterns first, and stands in a class instead. |
| Addition | [algorithm-addition.md](algorithm-addition.md) | A request with each column, a neuron for each case that can be asked, a function per case, and a carry the world shows while it teaches and the machine infers afterwards. |
| Copy | [algorithm-copy.md](algorithm-copy.md) | Shown a digit and asked, the machine says it: one parameter in the answer's table over the digit shown and the digit said, bound from what was shown, so a digit never said before is said the first time; once a class covers the shown digit the same parameter reads it through the class's value neuron. |
| Equal | [algorithm-equal.md](algorithm-equal.md) | Shown four digits in one frame and asked, the machine says the third where the first two are the same and the fourth where they differ: equality is a parameter over two slots of the request, the answer is a copy by a parameter over a slot of the request and the slot of the answer, bound from what was shown, and the request answers by a vote the one voter that knows the situation decides. |
| Hit | [algorithm-hit.md](algorithm-hit.md) | Offsets from where the world reports something approaching as the reference frame, a parameter that takes whatever comes, a value neuron and a lesson per thing, and a thing never seen answered by the habit of the event the world reports in common. |
| Loop | [algorithm-loop.md](algorithm-loop.md) | Shown a cue, the machine taps in each of the next three frames and then stops. There is no counter: each tap's table names what stood a frame back, so the states of the loop are three children each built over the one before, each connected to the one after it and the last to the event the world shows at the end. Stopping is a thing that follows and not an absence, since speaking falls through the cover; taught to stop on a cue instead of a count, the same machine is a while loop. |
| MNIST | [algorithm-mnist.md](algorithm-mnist.md) | An image channel with two spatial dimensions and a binary pixel dimension, 256 buckets or three channels for color later; patches at the base, odd shapes closed on the frontier, a digit action taught by demonstration. |
| Stocks | [algorithm-stocks.md](algorithm-stocks.md) | A price channel per instrument with own and disown as its actions, under a policy where an action is no one's neighbor and only events and patterns connect, to actions alone; patterns over frames; a world that runs the right action wherever the machine has none; and a reward scoped to the channel and the frame, each frame. |
| Text | [algorithm-text.md](algorithm-text.md) | A stream of letters at reach 1: at the base a class of the letters that stand before each letter, its value neurons the pairs; words of any length from children and letters on the frontier, a doubled letter as a parameter. |

# 10. Further reading

Each document is written in the terms of this specification and references its keys.

| Document | What it holds |
|---|---|
| [algorithm-remarks.md](algorithm-remarks.md) | The theorems, and the commentary on why the design is shaped this way, keyed by the same D and R numbers. |
| [algorithm-evaluation.md](algorithm-evaluation.md) | The falsifiable claim, the risks with what would be done about each, and the open questions. |
| [algorithm-implementation.md](algorithm-implementation.md) | The implementation plan: the per-neuron state and calls, the build stages, and the deltas against the current code. |
| [hippocampus.md](hippocampus.md) | An addendum: the second organ, running beside this machine over the same neurons, with rules keyed **H**. |