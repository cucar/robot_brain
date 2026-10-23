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
A function has a single child, which stands for every occurrence of it.
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
the machine holds is priced as a **file**, the dictionary of every neuron's patterns and the transcript the apex
would fill, and shortening that file is the first objective.

## 3.1 Neurons

> **D2 — Neuron.** A neuron is a symbol, and a type: its activations are its instances (D8). It is identified
> by its id. Ids are handed out in sequence, so a smaller id is an older neuron. It is a processor with a
> memory of its own, and that memory holds a patterns table (D32), a history (D18) and connections (D25). It
> is a base neuron (D52) or an interneuron (D53).

Older neurons have smaller ids. That's how ties are broken.

The machine holds an index of its neurons by id. Every pattern, connection, activation and deletion that names a
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

> **D53 — Interneuron.** A function's child, its **call neuron** (D48). The machine creates it the first time a
> bid carrying the function is accepted and no interneuron already stands for the same thing (R16, R43, D36).
> It has an id and no dimension and no channel. Its activation fires in the frame its bid was accepted in, at
> the bidder's coordinate.

An interneuron is not an event or an action, though what it stands for may be either. A child may have
several parents, since several functions may come to share it (R43). Its activation's coordinate is its
bidder's, and never an average over what it covers.

> **D36 — Interneuron index.** The machine's index of its interneurons by what each stands for, beside the one
> by id (D2) and the one by base symbol (D52). An interneuron stands for its expansion (R28): its function's
> nodes and relations (D34) with the variables left open, each node in them for what it stands for in turn, down to
> base symbols. No function names a neuron built over its
> own child (D27), so what a neuron stands for never contains itself and is finite. **No two interneurons
> stand for the same thing.** An accepted bid whose pattern has no child is wired to the interneuron that stands
> for what its child would, whether that neuron bid this frame, fired frames ago or has not fired within any
> open window, and a new one is created only where none does (R16, R43).
>
> For each interneuron the index keeps its definition and its parents. A parent is a neuron, and the index
> keeps which of its functions points at the interneuron. A neuron becomes a parent when update child points one
> of its functions at the interneuron (§4.1). It stops being one when that function detaches or retires.
>
> An interneuron's definition is set by an accepted bid. A bid sets it when the bid's function differs from the
> interneuron's definition, or the interneuron is new. In one frame only the first such bid the election accepted
> sets it. A later one is compared against that.

**A function is a pointer to a child, and a child is a neuron.** The two are made apart. A neuron adds a
**function** to its own table, which may enter a cover at once, and bids it with no child. The first time a
bid of that function is accepted the machine gives it a **child**, a neuron (D2): one it mints, or one another
function already has for the same activations (R16, R43). **Patterns stay with the neuron, and neurons belong
to the machine.**

> **D48 — Call neuron.** A function's child, one per function: the neuron the machine gives a function the
> first time a bid carrying it is accepted (R16), shared by every function the machine has wired to it (R43).
> It fires when a bid carrying the function is accepted, at the bidder's coordinate (D53), once, covering the
> bidder and the function's fixed nodes (§6.4); what stands in the function's variables is bound beside it, on the
> apex (D10). It says which function ran. Its connections are one set for every value its variables have taken, so a
> neuron above learns about the function regardless of its arguments by a pattern over the call neuron alone,
> and about one argument by a pattern over the call neuron and the activation bound in that variable. It holds a
> table, a history and connections of its own, and is expanded through the function whose bid fired it (R28).

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
> cover with multiple patterns (D28), so several of its children may be active at one coordinate, each at its
> bidder's (D53).

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
> or binds is strong, and weak when every one of them is: a situation made entirely of inferred inputs is
> inferred.

> **D37 — Call.** One activation of a call neuron (D48), together with its **arguments**: the activations bound
> in its function's variables on that occasion (D10, D38). It is made in one of two ways and is the same thing either
> way. **Recognized**: the election accepts a bid carrying the function, the call neuron fires at the bidder's
> coordinate (D53), as strong as what it covers (D40), and the arguments are what stood in the variables. **Inferred**:
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

## 3.5 Space

> **D4 — Reach.** The farthest a neuron sees, in each activation dimension (D1): its span plus the radius `R`
> of that dimension, in positions.
>
> | | |
> |---|---|
> | span of a base neuron | `0` |
> | span of an interneuron | the largest, over the patterns wired to it and over everything each names, of that node's offset from the owner plus the node's own neuron's span |
> | reach | the span plus `R`, per activation dimension |
>
> The span is how far the farthest base activation under the neuron lies from its activation's coordinate, read
> off its patterns as they stand: it moves when a pattern re-centers, when a pattern is wired to the child
> or retired, and at no other time. A neuron sees `R` beyond the farthest node it stands for. A base neuron
> stands for nothing but itself and sees `R`; a child over a pair of adjacent positions sees `R + 1`; a child
> whose coverage ends where another's begins sees that child's coordinate, one past its span, which is how two
> adjacent coverages are joined into one by a child above them. A child one letter longer than its parent's
> sees one position further. In space the reach runs either way; in time it runs back, and the same number
> runs forward as the activation's window (D9). The reach is a bound, not a distance: a neuron sees every
> distance up to it, exact within `R` and rounded to one significant digit beyond when written (D6). `reach_t`
> is this reach in the time dimension, the neuron's window.

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
> of the box, and an activation is not its own neighbor there. Above the base several children promoted at one
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

## 3.6 Apex

> **D49 — Apex.** The activations of a frame that no accepted bid covers, base neurons and interneurons alike:
> a base neuron no pattern found worth naming stands in it beside a child many patterns deep. It is the machine's whole state at
> that frame, and it has four names for the four things it is. It is the **frontier**, looking back: everything
> underneath it is recovered by expanding it, so it is what the file counts. It is the **situation**: what
> every neuron sees as its neighborhood (D5), what connects to what ran, and what votes. It is the
> **wavefront** of recognition: where the frame's levels stopped (R26), and from where the wave runs
> forward, each activation of it reading its own connections ahead, which it can do only because it is also
> the frontier, the neighborhood every neuron was built on. And it is the apex, the top of the hierarchy as it
> stands. The four are one set, and coverage takes a neuron out of it in every role at once.

> **D10 — Inhibition.** Coverage does three things to an activation, all from the frame it is covered in. It
> leaves the file: the file counts its coverer's call, not it. It is not heard: resolution reads its coverer's
> standing inferences, not its own (R46). And it stops connecting: from the frame after it is covered it writes
> nothing about what follows.
>
> **The last connection.** In the frame it is covered, the activation connects once more, to the neuron that
> covered it, the function's child. The connection is at the
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
>
> **Bound is half covered.** An activation standing in a variable of an accepted bid (D38) is **bound**. The call
> covers its position: the function's relation says where it stands, so its position is no longer in the file.
> Its neuron is not covered: it is written as it was, a neuron (D12, D13). It stays on the apex, it is a
> neighbor, it is heard, and it connects to what follows. A call hides what is the same on every occasion, its
> owner and its fixed nodes, and leaves visible what varies. A parameter binds each of its places the same way.

## 3.7 File

> **D12 — File.** Two parts, both spanning the whole run. **The dictionary**: every neuron's table, its
> functions and the classes they use (D38). A function's body is written there once, and nowhere else.
> **The transcript**: the apex of every frame (D49), each activation as its **record**:
>
> | Entry | Its record |
> |---|---|
> | a bare neuron | present; the neuron; its position |
> | a call | present; its child; its position |
> | an activation bound in a variable | present; the neuron; its position, less what the call fixes: the function's relation says where it stands |
> | an absence | absent; the neuron a call placed that did not stand; its position |
>
> Pictured as text, the dictionary is a line per pattern and the transcript a list of entries.

**A variable covers a position, and nothing else.** The call fixes where each of its variables stands, by the
function's relations, so a value's position is not written again. Which neuron stood there is written as it
was, a neuron. A call compresses its function's body: its owner and its fixed nodes are written in the
dictionary once, and each call writes none of them. The values in its variables stay on the apex and in the
transcript.

**An error is an absence.** A call places every node its function names. A node that did not stand is an entry
of its own, at the coordinate where it should have stood: absent, and the neuron. The decoder expands every
call, then removes every neuron an absence names. Whatever stood there instead is on the apex, uncovered, and
is written as itself. So a wrong neuron costs an absence and a neuron, and nothing there costs an absence alone.

**Both types are in it.** An action the machine executed is a neuron that fired (D8), and it stands in the
file exactly as an observed event does, compressed by the same patterns (D5).

**It holds nothing about the future.** A neuron's connections — what followed its activations, event and
action alike, and what the actions earned — are in neither part (D25), and neither is an inference that did
not run (R32).

**It holds nothing about the search either.** Populations, estimates and margins are the machine's and
never the file's, because expanding an apex neuron needs the patterns and nothing else.

**It holds the values.** What stands in a variable is written as its neuron (D13), its position covered by the
call (D10), and expansion puts it back at its place (R28).

## 3.8 Prices

> **D13 — Field.** Every cost in the design is part of the file, counted in **bits**, and nothing is ever
> written: the file is the yardstick (§1.1). Each thing the file would hold is a **field**, a choice among the
> options of its own alphabet, and a field costs `log₂` of its alphabet's size however often it is used: a
> fixed-length code, with one alphabet per kind of field.
>
> | Field | Cost | The alphabet |
> |---|---|---|
> | a neuron | `log₂ n` | the `n` neurons the machine holds, base neurons and interneurons alike |
> | a position | per activation dimension, summed | in time, the offsets back to the previous entry of the transcript, `offsets(r) + 1` at the activation's neuron's reach `r` in time (D4, D6); in any other dimension, the positions of the layout the activation stands in (D3) |
> | an offset | `log₂ b` per dimension, summed | the `b` buckets of that dimension at the owner's reach `r` (D6): `offsets(r) + 1` in time and `2 · offsets(r) + 1` in any other |
> | which class | `log₂ \|classes\|` | the owner's classes, one of which a variable uses; nothing with one class |
> | present or absent | `1` | the two kinds of entry (D12) |

`n` is the machine's count of its neurons, the base neurons and every neuron above them together. It starts
at the count of base neurons the machine declares (D1). It is raised when a neuron is created and lowered when
one is deleted, and the machine hands it to every neuron with its call (R20). **Every neuron costs the same to
name**, a base neuron as much as an interneuron. Every other alphabet is a table's own: its classes and
their members.

**The error correction** is an absence per node a call placed that did not stand (D12): present or absent, a
neuron and a position, the same fields as the activation that failed to be there. A call where every node held
pays nothing to say so. A call where every node failed pays an absence per node and covers only its bidder, so
its savings are negative and it never fits (D22).

**An activation costs what its record takes to state** (D12), and it is the same wherever the activation is
counted:

| Activation | Its fields |
|---|---|
| a base activation | present or absent; a neuron; its position |
| a call | present or absent; a neuron, its child; its position |
| an activation bound in a variable | present or absent; a neuron; its position less what the call fixes |
| an absence | present or absent; a neuron; its position |

The machine keeps the cost with the activation, and a saved neighborhood keeps it with each neighbor (D18).

**Covering an activation saves exactly what it cost.** So a child that only renames what it covers saves
nothing. A bound activation costs its neuron, and covering it later saves its neuron and no more.

**Each kind pays for what it says.** A function says which nodes and the offsets between them, once, and
each occurrence says only that it happened: its child and its position, with an absence per node that did not
hold. Where it stood is
said once, as the call's position, and every node it names stands at its offset from there. So a call covers
the positions of everything it names, its variables' included. A class says which set may stand at a place,
once. A parameter says which places agree, once. Neither writes which neuron stood there: that is written as
it was.

**Where an activation stands is written.** The transcript lists what stood, and silence is what it does not
list, so each entry says where it stood, in every activation dimension. Time is written as an offset back to
the previous entry, in the alphabet patterns use for time (D6). Every other activation dimension is written as
a position in the layout. How many things stand at one place, or follow one instance, needs no field: each is
an entry of its own. **Nothing
about frequency enters**: a field costs the same however often it is used.

## 3.9 File length

The file's length `L` is two sums (D13): over every neuron's table, what its functions and classes cost, which
is the dictionary; and over the activations on the frontier, what each costs, which is the transcript. There is
one reading of it. A call is its child and its position, with an absence per node it placed that did not
stand, and what the call fired stands on the frontier at exactly that. What it bound stands on the frontier as its neuron, its position covered. A
neuron above counts a child at what it cost below, so nothing is saved by naming it again and nothing is gained
by counting it twice. The election takes activations off the frontier at their cost, and puts a call in their
place at its price (R22). A neuron's margin does the same over its own history (D30). Nothing computes `L`: every quantity the design uses
is a **difference**, which is finite however long the run is.

## 3.10 Parameters

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
| create neuron           | The machine creates it when an accepted bid needs one and none can be reused, holding nothing: a call neuron for a function. | R16, R43 |
| process functions       | At its level: everything structural for the activations that fired this frame.                                                                             | §5, R20    |
| update child            | The machine calls a neuron with three things: one of its functions, that function's child, and that interneuron's definition. If the definition is the function, the neuron points the function at the child. Otherwise the neuron either adapts the function to the definition or detaches the function from the child, whichever leaves the larger margin. It returns what it did. The machine uses that to keep the child's parents in the interneuron index. | D36, R16, R43 |
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
> its history, fits the frame's rows, re-centers what they changed, builds every candidate that pays, merges
> what became identical, retires what no longer pays, and returns the bids of each activation together with everything it retired
> (R20).

## 4.2 History

> **D18 — History.** A neuron's record of its own activations: the last `H` of them, oldest first, each at its
> coordinate, holding the neighborhood it observed in the frame it fired (D7), each neighbor at its exact
> offset (D6) and with what it cost, and the activation's own cost beside them, since a function
> covers the activation too (D13, D22). It is the evidence every structural decision
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

> **D32 — Table.** A neuron holds one **patterns table**, **the table** for short: its functions (D38), and the
> classes its functions use as variables (D41), each with its members. Fit reads it (D54); adding, retiring
> and merging write it (R15, R18, D43).

> **D16 — Node.** What a function names: a neuron (D2), a variable (D38), or the owner's own activation, which
> stands at offset zero in every neighborhood the owner records (D26).

> **D34 — Relation.** Two ends and the offset between them (D6), one component per activation dimension the
> two share. It is of one of three kinds, by what stands at the ends:
>
> | Kind | Ends | The part of |
> |---|---|---|
> | node to node | two nodes (D16) | a function's fixed structure: a set of these |
> | same to same | one and the same neuron at both, whichever it is | a parameter: the places it ties |
> | node to open | a node at one end, and at the other an offset where different neurons stand, never two in one row | a class: the place it is named at |
>
> In a pattern the offset is written rounded, as every written offset is, and the relation holds where its ends
> stand at any exact offset in the written offset's group. Relations compose: given the offset from `a` to `b`
> and from `b` to `c`, the offset from `a` to `c` follows, so a set of relations that connects its nodes fixes
> every other relation among them.

> **D15 — Pattern.** What a neuron **names**: a **function**, a set of nodes (D16) connected by relations (D34),
> one relation fewer than nodes, the rest following. A function fits inside the box a neighborhood spans (D5).
> Every function names the owner's activation among its nodes, so it holds at the owner and nowhere else:
> where the activations around the owner satisfy every relation it writes. A function is the collapse of the
> rows it fits (D27), moves as they move (D29), and promotes one child, a call neuron (R16). Its `id` is its
> creation order within its owner's table, a handle that survives re-centering. It breaks one tie: where two of
> the owner's functions measure the same in a row's cost (D30), the older `id` is taken. The election never
> reads it.
>
> **A pattern is not a neighborhood.** A neighborhood is what one activation saw where it fired, saved once and
> evicted `H` activations later; a pattern is what the neuron claims. A neighborhood is a fact, a pattern a claim.
> Neither is a frame: a frame is one column, either of these the whole window. What followed is in neither, and
> is held as the neuron's connections (D25).
>
> **A pattern is written from the owner.** Its relations run from the owner's activation to the nodes it
> names, so where it stood is where the owner stood, and that is never written (D13): the bidder's coordinate
> says it, and the child fires there (D53).
>
> **Its entry in the table** costs a neuron per fixed node, which class per class, and an offset per relation it
> holds (D13). The owner's activation costs nothing there, the neuron being the table's own.

**A function's body is its nodes with the offsets between them.** Some nodes are fixed, a particular neuron at
a place; some are **variables**, places where the function is general. A variable is where the function fits more
activations than any one neuron there would let it: it names the place and leaves open what stands in it. What
stands in a variable on one occasion is the variable's **value**, and the values of a call's variables are its arguments
(D37). **Classes are a function's local variables and parameters are its arguments**: a class is a place
whose content is decided where the function runs, among what has stood there before, and a parameter is
one value handed in and used at every one of its places. A variable exists only inside a function, is never bid
alone, and promotes no child: what stands in it stays on the apex, bound (D10).

> **D38 — Variables.** A function's nodes are its owner's activation, fixed nodes, each a neuron at an offset
> (D26), and variables. A variable leaves open what stands at a place, and is of one of two types:
>
> | Variable | Fixed, priced once | Open, priced on each occurrence | What it covers |
> |---|---|---|---|
> | a class (D41) | which class of the table: a set of neurons, its members | nothing: the member is written as itself | its position; the member stands bound (D10) |
> | a parameter (D44) | which places of the function agree, as same-to-same relations | nothing: the neuron is written as itself at each place | the position at each place; the neuron stands bound at each (D10) |
>
> **The two types are closed in opposite ways.** A class is closed on its members and open on where: a
> neuron that is not a member does not fit, and any function of the table may use the class at any place. A
> parameter is open on its members and closed on where: any neuron fits, seen before or not, and only
> where it stands at every place the variable ties. A function may name no fixed node beside the owner, and be the
> owner with variables alone; one over the owner and one parameter is the function that parameter is handed
> to.
>
> **Every function names the owner's activation.** It stands at offset zero in every neighborhood (D26), so a
> function holds at the owner and covers it: the bidder and the fixed nodes are covered together, and what
> stands in the variables is bound (D10). On every occurrence a bid is priced as its child and its
> position, with an absence per node that did not hold (D13, D22). A function holds nothing
> but its nodes and relations: no branch, no loop. What branches is which function fires; what loops is the
> frame; what a variable held is on the apex, bound beside the call neuron.

> **D41 — Class.** A set of neurons the table holds, its **members**, and nothing else: **one of these stands
> here**. In the table it costs a neuron per member (D13). A function uses it as a variable at a place (D38); several functions of the table may use one class, each
> at its own place, and a function may use one class at several places, a member bound at each. Members stand
> in place of one another: two neurons that have stood together where the class is used are never both members.
>
> **Its value**, at a place, is the member that stood there, which stays on the apex, bound (D10). Where no
> member stands at a place the class is used, that is an absence (D22), and the variable does not hold there.
>
> **A class is not a neuron.** It is made by a function's collapse when a place pays as a class and no
> class of the table fits it (D27), grows and shrinks on its members by the collapse of any function that uses
> it, and is deleted when no function uses it or it is left with one member (R41). A member joins when what it
> covers, wherever the class is used, pays for its entry, and leaves when it no longer does; so what is learned
> about a member at one place holds at every other.

> **D44 — Parameter.** A variable of one function that ties two or more of its **places** together: **the same one
> stands at all of these places**, whichever it is. It is written as same-to-same relations between the places
> (D34), one fewer than places, and the relations compose. Beside it the function keeps its **members**, the
> neurons that have stood at all of its places within the history. The list is a record and not a condition. It
> says nothing about what may stand, and it prices nothing (D13). In the table a parameter costs nothing beyond
> its relations.
>
> **Its value** is the neuron that stood at its places. It stays on the apex, bound, at each of them, written as
> itself (D10).
> Where the places disagree, the value is the neuron at the earliest of its places, the smaller position on a
> tie, and every place that holds something else is an absence (D22).
>
> **It is open on its members and closed on where.** Any neuron fits, seen before or not, and only where it
> stands at every place: a thing that comes for the first time is the parameter's value the first time, and is a
> member from then on (D27). A place joins the variable when it comes to repeat the value, and leaves when it stops;
> a parameter left with one place is a fixed node or nothing (R41).
>
> **Bound from what has happened, placed at what has not.** When a function is expanded with no argument for a
> parameter (R28), and the variable's places fall partly in frames that have happened and partly in frames to
> come, it takes its value from what stands at the first and places it at the second. That is how a value is
> carried forward in time, and it is the whole of copying.

## 4.4 Fit

A neuron reads its table against its rows. This section defines what it finds there.

> **D54 — Fit.** A function fits a row where its savings there are positive (D22): what its nodes that hold cost
> is more than its record and its absences. A row is read as it stands: each neighbor is the neuron at its
> offset, a child on the apex is a neuron as itself, and the owner's activation is itself at offset zero.
> Nothing in a row is reserved: two functions of one table may fit one row on the same neighbors.
>
> **Uses.** The rows a function fits are its **uses**. They are what the collapse reads (D27), what re-centering
> watches (D29), and where the function is bid (D31).
>
> **Holding and failing.** Measured against a row, each node a function names holds or fails:
>
> | Node | Holds where |
> |---|---|
> | the owner's activation | always |
> | a fixed node | its neuron stands at its offset, or a child standing on the apex stands for it there (D22) |
> | a class | a member stands at its place; it takes members only (D27) |
> | a parameter | its value stands at the place; it takes any neuron, and one that was not a member is one from that row on |
>
> A node that fails is an absence (D12). Where it would have stood follows from its relation to any node that
> holds.

## 4.5 Savings

> **D22 — Savings.** What one function `p` saves over one row `O`: what it covers there, less what it costs
> there, the fields of its occurrence (D13).
>
> | | |
> |---|---|
> | coverage | the owner's activation and each fixed node that holds, a neuron and a position each; and the position of each place a variable holds at |
> | price | its record (D12): its child and its position; and an absence per node that did not hold |
> | savings | coverage less price |
>
> A neighbor in a variable counts toward the coverage at its position only: the call covers where it stands,
> and its neuron is still written.
>
> What a function names holds or fails: a fixed node fails when its neuron is absent, a class where no
> member stands at its place (D41), a parameter at each place where what stands differs from its value
> (D44), and the owner's activation never, since it is there wherever the owner reads.
>
> **A child holds a node it stands for, and is not covered by it.** Where a function names a neuron at a
> place, and a child standing on the apex stands for that neuron there (D47), the node holds. The function is
> credited nothing for it, and the child is not covered. The child's record already writes that neuron, and the
> call's record does not write the child. So a match inside a child is a condition the function meets, and a
> function saves only what it takes off the apex.

**What a function fires on the apex stands in for what it covers**; that is what it saves, and that is the
coverage. What it costs is its record. The owner's activation is on the coverage side and the child on the price
side, and where the owner is a base neuron each costs a neuron and a position, so the two cancel. On the board
a bidder is credited once, to the first bid of it accepted (R24); after that the child is on the price side
alone. A neuron the function does not name is not in the account at all: it costs what it cost whether the
function exists or not.

**This is the only valuation in the design**, and it is read over three different sets, so the three numbers
differ, and are meant to:

| Read by                               | Over                  | Against                                  | Settles                            |
|---------------------------------------|-----------------------|------------------------------------------|------------------------------------|
| a neuron, in its margin (D28, D30)    | its functions         | each row of its history                  | its table                          |
| the machine, in the election (R22, R24) | this frame's bids   | the board                                | what is credited, and so the file  |
| the machine, in the match (R49)       | one inferred function | the frames it reaches back into          | whether the inference is proposed  |

## 4.6 Greedy cover

One rule decides what a set of bids writes for a set of activations. The election runs it over the board, and
a neuron runs it over a row to price its table (D30). It is stated here and cited from both.

> **D28 — Greedy cover.** The operation that covers a set of activations with **functions**, each read against
> one of those activations, and pays each covered neuron to exactly one function. It repeats:
>
> 1. **Measure** every function not yet taken against the still-uncovered neurons of its activation: its coverage
>    is what the still-uncovered neurons it covers cost, the activation itself among them, and its price is the
>    fields of its occurrence there (D22).
> 2. **Take** the one with the highest `coverage / price` if `coverage > price`, and **stop** otherwise. It is
>    paid for the neurons it was measured on. Those it covers leave the uncovered set. Those it binds stay in it
>    at their neuron's cost, since only their position is covered, so a later round may cover them (D10). The
>    round repeats from 1.
>
> **What it returns is what was written and by whom**: the functions taken, what each was paid for, and what no
> round took, which stands as itself. `price` is fixed by what fired and cannot change between rounds;
> `coverage` only falls.

**The two callers differ in what they cover and in how they break ties, in nothing else.** Each states its
tie-break where it runs (D15, R24).

| caller                     | functions | to cover                      |
|----------------------------|---|-------------------------------|
| neuron - margin (D30) | the functions of its table that fit the row (D54) | the row, whole; priced and kept nowhere |
| machine - election (R24)  | this frame's bids (D31) | the board (§6.2) |

## 4.7 Bid

> **D31 — Bid.** What an activation offers the machine: every function of its neuron's table that fits its
> neighborhood (D54). Competing explanations of one
> neighborhood are all bid, and the election picks among them (R24).
>
> **A function is bid whole.** The machine takes it or leaves it, its variables with it: it cannot hold where they
> do not.
>
> **A bid carries its function, and the machine finds the rest.** The function's relations say where each node
> stands from the bidder, so the machine reads what stands in each variable off the board itself (R22). An accepted bid covers
> its bidder and its fixed nodes, the bidder credited once, to the first bid of that activation the election
> accepts (R24), and binds what stands in its variables (D10).
>
> | Component   | Description                                                                      |
> |-------------|----------------------------------------------------------------------------------|
> | function    | the id of the function bid; an id is its creation order (D15) |
> | relations   | the function's relations and variables, as they stand in the table (D12) |
> | child       | the id of the child the function promotes; none until a bid of it has been accepted (R16) |
>
> **Resolving a bid.** The machine places the owner at the bidder's coordinate and each place at its offset
> from it. A coarse offset spans a group of exact offsets (D6), and the place is read over every activation in
> the group. A fixed node holds where its neuron stands there, or a child standing for it (D22). A class place
> holds where a member stands there. A parameter's value is the neuron standing at the earliest of its places,
> the smaller position on a tie, and it holds at each place where that neuron stands (D44). Where several activations could
> answer one place, the one not yet covered is taken first, then the one nearest the owner, then the earlier
> coordinate (R24). What stands in each variable on that occasion is the call's arguments (D37). Expansion
> reads a call back by the same rule (R28).

## 4.8 Relations

> **D47 — Relation index.** What a neuron counts over its history (D18): every relation (D34) its rows hold,
> each counted by the rows that hold it, owned or not. The ends are neighbors at their coordinates and the
> owner's activation (D16). The index is updated when a row is admitted and when a row is evicted. Each kind of
> relation is the evidence for one type of pattern:
>
> | Kind | What stands | Counted by | Evidence for |
> |---|---|---|---|
> | node to node | `a` and `b` at `δ` from each other | the rows that hold the pair at `δ` | a function of the two, or a fixed node joining one |
> | same to same | `n` at `c`, and `n` at `c + δ` | the rows that hold one neuron at both ends, whichever neuron | a parameter tying the two places |
> | node to open | `a` at `δ` from `c` in one row, `b` at `δ` from `c` in another, never both in one row | the neighbors holding either there, `a` and `b` being the two that have stood there most | a class at `δ` from `c`, of `a` and `b` (D41) |
>
> The first and the third hold between nodes of any kind, the owner's activation among them.
>
> **A child stands for what is under it.** A child on the apex is a neighbor as itself, and as each of the base
> symbols it stands for, each at its own offset (R28). As itself it costs what it cost. As a base symbol under it
> it costs nothing: that symbol is written in the child's record. So a relation through a base symbol under a
> child says where a function holds, and never what it would save (D22).

The owner stands in every row at offset zero, so one neighbor that keeps standing at one offset from it is a
node-to-node relation with the owner, and a function of the two. At the base, in a stream of text, that is what
every row carries.

How the index is laid out is the implementation's. What it holds is not: a relation's count is the number of
rows of the history that hold it.

## 4.9 Collapse

> **D27 — Collapse.** The operation that grows a seed (D33) into a function, and re-decides what a function
> names once it exists. It reads the function's **uses** (D54), whole rows, and everything the function names
> is decided by its price over those uses (D22), from the uses alone. What a neighbor is worth is always what it
> cost (D13), read from the saved neighborhood.
>
> **The uses.** For a function of the table, the uses are the rows it fits as it stands before the collapse
> (D54). For a candidate, they are the rows that hold its seed (D33). Each use is read as it stands: every
> neighbor is the neuron at its offset, a child on the apex is a neuron as itself, a base symbol under a child
> holds a node there and is worth nothing (D47), and the owner's activation is itself at offset zero. Nothing
> in a use is skipped because another function names it. Two functions of one table may read the same neighbor
> in the same row and both name it; which of them is paid for it is settled in the margin (D30) and on the
> board (R24), never here.
>
> **A place.** A place is an offset `δ` from a node already in the function, and the collapse decides what to
> name there, over the uses, among four options:
>
> | Option | Covers, in each use where it holds | Holds where | Entry |
> |---|---|---|---|
> | a fixed node `x` | what `x` costs: a neuron and a position | `x` stands at `δ` | a neuron and an offset |
> | a class | the position | a member stands at `δ` | which class, plus the members a new class names |
> | a parameter tying `δ` to a place already in | the position | the same neuron stands at both | an offset |
> | nothing | nothing | everywhere | nothing |
>
> The option taken is the one with the largest value of what it covers over the uses, less an absence (D13) for
> each use where it does not hold, less its entry. A node that holds in few uses is worth less than nothing.
> A class's members are decided with it: a candidate member is a neuron that has stood at the place and
> never in a use together with a member there, counted wherever the class is used, and the candidates are
> taken in order of that count, each while what it covers beats a neuron's entry and the wider choice it gives
> every use the class already has, `uses · log₂(|K′| / |K|)`, `K′` being the members with it added. The first
> that fails ends it. A member leaves by the same test read the other way. A class of the table with those
> members is used rather than a new one (D43).
>
> A parameter's members are a record of what has stood at all its places, and its places are decided one
> by one: a place stays tied while what it covers over the uses beats the failures where it disagrees and an
> offset in the table; one that comes to repeat the value joins, and one that stops leaves (D44).
>
> Every place is decided independently, from the input alone, so the result does not depend on the order
> places are decided in, and a node may enter by its relation to any node already in, since relations compose
> and the function is the same whichever is written (D34).
>
> **After the collapse** the function's fit is re-read over every row (D54). Where its uses changed, it
> re-centers again at the next call (D29). One collapse per function per call.
>
> **A function never names a neuron built over its own child.** A neuron is built over a child when what it
> stands for passes through that child (D36): a function over the child, or over a neuron built over it. A
> function with a child skips such a neuron as a fixed node and as a member of any of its variables. So no neuron
> stands for something that contains itself, and every expansion reaches the base.

The collapse is the only operation that decides what a pattern names. It runs in two places:

- **Re-centering** (§4.10, D29) collapses a function of the table over its uses.
- **Creating** (§5.4, D33) collapses a seed into a function, over the rows that hold the seed.

**Competition is what keeps patterns apart.** Two functions that come to name the same neighbors are priced
against each other in the margin (D30). The one that explains no row best falls below its entry and is retired
(R18), and identical ones merge (D43).

**Every place is decided by the same uses.** A row says something at every offset from the owner, a neuron or
a silence, so the outermost place is decided over the same `s` as the one beside the owner. Each function's collapse has its own `s`; nothing is pooled across functions, and no threshold,
smoothing or probability estimate enters.

## 4.10 Re-centering

> **D29 — Re-centering.** The operation that re-decides what a function of the table names (D15) once its uses
> have changed (D54): a use evicted when the history is full (D18), or an admitted row it fits. It runs the
> collapse (D27) over the uses as they stand. A place may widen into a variable or narrow to a fixed node here,
> and the uses widen or narrow with it. It never takes a neuron built over the function's own child (D27).

## 4.11 Margin

> **D30 — Margin.** What a function is worth over the history, less its entry in the table (D15). It is priced
> against what the table does without it, never against nothing.
>
> | | |
> |---|---|
> | the cost of a row | what the election would write for it from this table: the greedy cover (D28) of the row by the functions that fit it (D54), each taken function at its price there, and what no function covers at its own cost |
> | benefit of `f` | the sum, over the rows of the history, of the row's cost without `f` in the table less its cost with `f` |
> | cost of `f` | its entry in the table (D15) |
> | margin of `f` | benefit less cost |
>
> **Without `f`**, the row is written by the rest of the table, so another function may take what `f` held. A
> function is worth what it adds over the best the table does without it. Where another function does better
> in a row, `f` is worth nothing there. A class has no margin of its own: it is worth what the functions using
> it are worth with it in them, and it goes when none uses it (R41).
>
> **Both costs are computed when the margin is asked, and kept nowhere.** The neuron holds no cover and no
> owner. This is the one place the election's arithmetic runs inside a neuron, and it is a price.
>
> It is the difference in the file (§3.9) between the file with the function and the file without it, and both
> terms are sums over what the neuron already holds.

A pattern is added only when its margin is strictly positive (R15) and retired only when strictly negative (R18).

## 4.12 Seeds

> **D33 — Seed.** A relation of the index (D47) that no function of the table writes between the same nodes at
> the same offset, counted in two or more rows. A relation a function writes is that function's: re-centering
> is where it grows (D29). A relation outside every function starts a new one, and that includes a relation
> lying inside ground an existing function fits.
>
> | Relation | Grows into |
> |---|---|
> | node to node | a function over the two nodes, the owner among them where it is one |
> | same to same | a function over the owner and a parameter tying the two places |
> | node to open | a function over the owner and a class at the place |
>
> Every seed is tried, once per call, in this order:
>
> 1. **Order** the seeds by what each would save as it stands: D22's savings of the relation taken as a whole
>    function, summed over the rows that hold it. Ties go to the higher count, then to the older neuron,
>    declaration order at the base and creation order above it, then to the shorter offset.
> 2. **Grow** each by the collapse (D27) over the rows that hold it, which decides every other place, fixed,
>    variable or nothing.
> 3. **Keep** it if its margin is positive (D30, R15). A kept one is in the table for every seed after it.
>
> Nothing is consumed, so nothing shrinks and nothing stops early. It returns the functions added, and the
> classes they made.

A node-to-open relation whose members do not pay as a class grows a function over the owner and the
neuron that stood there most, as a fixed node, if that pays.

> **D43 — Merging.** Whenever a function is added or re-centered, it is compared with its table. Of each group
> of identical functions the oldest is kept, and the rest are retired (§5.5). Two classes of one table with the
> same members are one class, and the functions using the newer use the older.
>
> | Kind | Identical when | The kept one takes |
> |---|---|---|
> | function | the same nodes, variables and relations | the others' child, where it has none (§5.5) |
> | class | the same members | the others' places in the functions that use them |

Relations compose (D34), so two functions that write different relations over the same nodes, but fix the same
offsets among them, are the same function.
Which children the kept pattern holds is in §5.5.

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

# 5. Process functions call

> **R20 — Call, in order.** Once per level it fires at, the machine asks one neuron for everything it owes
> that frame, handing it the count `n` of its neurons (D13) and **every activation of it that fired
> this frame with a neighborhood that is not empty**,
> each with its neighborhood, at age 0 (R1). They are processed together. A neuron none of whose activations
> has a neighbor is not called; such an activation is still open (D9) and on the apex (R27) like any uncovered
> activation.
>
> | Step            | Description                                                                                                   |
> |-----------------|---------------------------------------------------------------------------------------------------------------|
> | refresh history | Evict the oldest rows as the frame needs, admit the frame's rows, and update the index (D47).                   |
> | fit             | Match every function of the table against each admitted row (D54).                                             |
> | re-center       | Collapse every function whose uses changed over its uses, and re-read its fit (D29).                           |
> | create          | Grow every seed by the collapse, best first, and add each whose margin is positive (D33, R15).                  |
> | merge           | Compare everything added or re-centered this call against the table, exactly; keep the oldest of each identical group and retire the rest (D43). |
> | retire          | Price every function's margin against the table as it now stands, and retire every negative one, and every class no function uses or left with one member (R18). |
> | return          | The bids of each of the frame's activations, every function that fits its row; and everything retired.         |

The call runs before the election (R24).

## 5.1 Refresh history

The history holds `H` activations (D18). A frame bringing `A` activations evicts the `A` oldest once the
history is full, and fewer before, as many as it takes to make room. The frame's activations then join the
history, whole (D7). The index follows the rows: every relation in an admitted row counts one more, and in an
evicted row one less (D47). Nothing else touches it. A frame bringing more activations of one
neuron than `H` fills the history with the first `H` of them, in the order the input lays them out, and the rest
are not recorded; the machine reports it. `H` is the designer's to pick (R4), and a report says it was picked
below the most activations a neuron can have in one frame.

## 5.2 Fit

Every function of the table is matched against each admitted row, and the rows it fits join its uses (D54).
Nothing is chosen here. There is no cover, and nothing in a row is reserved for any function: a function is
either in a row or it is not. The election sees every function that fits the frame's row (D31,
§5.7), and it could never accept one that does not, since on the board a bid covers no more than its
neighborhood holds (R22).

**Cold start is silence.** A function with no uses names nothing. A neuron with an empty table bids nothing.

## 5.3 Re-center

Every function whose uses changed this call, by an eviction or an admission, collapses over its uses (D29), and
its fit is re-read over every row. A function with a node that holds nowhere still has uses, since the absence
is paid and the savings may stay positive, so it re-centers rather than standing still.

## 5.4 Create

Every seed is grown by the collapse and priced by its margin, best first (D33). Each that pays joins the table.

> **R14 — Candidate.** A candidate `C` is what one seed grows into (D33): the collapse of the seed over the rows
> that hold it (D27). Nothing seeds it from outside, and nothing grows it a neighbor at a time. The seed is in
> every one of those rows and is counted in at least two, so **nothing is ever built on a single occurrence**.
> Everything else `C` names holds in enough of those rows to pay for itself. **A candidate is built on whole
> rows**: a neighbor another function names is read like any other, since `C` competes for it in the margin
> (D30). The seed chooses the rows, and the rows decide every node.
>
> **Only what a pattern names is built, because that is all a pattern is.** What the child will be followed by
> is the child's own connections, formed by the child's own activations once it exists (D25). Nothing about it is
> decided here and nothing about it is priced.
>
> **The same history yields the same `C`.** The index counts whole rows (D47), so it is a function of the ring
> alone.
>
> **What `C` is worth** is its margin (D30): over each row, the row's cost without `C` less its cost with `C`,
> where that is positive. A function that explains a row better than the one that got there first is paid for
> it there, and the other is not.
>
> **What it is not.** It finds a local best and not the best `C` — choosing the pattern set is the
> facility-location problem and is not solvable exactly at any useful size. What it has instead is no free
> choice anywhere in it.

> **R15 — Add test.** The margin (D30) of a candidate: it is committed when its benefit over the history exceeds
> its entry in the table (D15). It then joins the table, its uses are the rows it fits (D54), and it is in the
> table for every seed tried after it and for every margin priced after it.
>
> **An accepted add shortens the file**, against the table it was priced on.
>
> **The test is offline and complete.** Every activation in the ring has a whole neighborhood, so the question
> is asked over the same evidence the margin will use when `C` is priced again.
>
> **Every seed is tried** (D33). A seed whose candidate did not pay is tried again at the next call.

> **R16 — What a child is at birth.** The neuron proposes, the election decides, and the machine creates. A
> function is the neuron's own: it is added, used in covers and bid on the neuron's evidence alone. A function
> with no child is bid as it is, and the bid asks for one, a call neuron (D48), if it is accepted (D31). **The
> machine gives a child only to an accepted bid**, so no neuron exists for a pattern the board never bought, and
> it reuses the interneuron that already stands for what the child would (D36, R43) before it creates one. A
> new child has an id and no dimension (D2). It is created with
> **an empty table**: its own patterns are over what it will see, which it has not observed yet. Its *existence*
> is decided by the election, its *structure* by itself.
>
> **A neuron may hold many children, and they do not contend.** Each covers the part of an activation its
> function owns, and several of them may be promoted at one coordinate (D8). What they share is a parent and a
> coordinate, not a place on the board. **A child may have many parents**: every neuron with a function
> pointing at it promotes it, each from its own position (R43).
>
> **Release is the reverse**: the parent's function retires, the machine reclaims. The retired patterns go on the return
> (R20). A retired pattern that never had a child leaves nothing to reclaim, and one whose child other patterns
> still promote leaves the child alone (R38).
>
> **Creating and updating are separate, and only updating waits.** The machine creates or chooses the child the
> moment the bid is accepted — an id, its parent and an empty table — which is all the frame needs to activate
> it, at the coordinate the machine places its activation at (D53), and call it (R17). Once the frame's last
> level has run, it calls update child for every function whose bid won a child, and for every parent of an
> interneuron whose definition changed (§4.1, §6). After the frame's connect and infer calls it reclaims every
> retired pattern's child now due (R38). Nothing in the frame reads the pointer except a later bid, so **the
> wait costs nothing**.

> **R17 — A pattern added in a call is bid in it.** The candidate joins the parent's table in the call (R15), so
> every activation of the frame whose row it fits bids it (R20, D31), with no child, and the bid competes like any other. If it wins, the machine gives the function a child, reused or new
> (R43), activates it at the next level and calls it there. A new child records its first neighborhood there and
> nothing more: its table is empty, and a candidate needs two neighborhoods (R14). If it loses, there is no child; the pattern stays in the
> table on its own margin (D30) and is bid again whenever it is in a cover.
>
> **It is born holding nothing.** No patterns, no history and no connections (R16). Everything it comes to
> hold is over the situations its parent's pattern actually took (D25), from its first activation on.

> **R41 — Life of a variable.** A **class** is **made** by a function's collapse when a place pays as a class
> and no class of the table has those members (D27, D43); **changed** by the member test of any function's
> collapse that uses it, once per call; **dead** when no function of the table uses it, or when it is left with
> one member, which is an ordinary neuron: it leaves the table, and the places that used it are re-decided. A
> **parameter** belongs to its function: **made** when the collapse ties two places (D27), **changed** by
> its place test at every call, **dead** when fewer than two places remain or its function retires.

## 5.5 Merge

Re-centering and creation can each leave two patterns in the table that are the same: two
functions with the same nodes, variables and relations, or two classes with the same members.
Merging compares everything added or re-centered this call against its own table, exactly (D43), and keeps
the oldest of each identical group, the lowest creation order, and with it the oldest child the group holds,
because it has accumulated a history, connections and estimates that a newer one has not.
A newer neuron with a history is never dropped for an older function with none. The rest are retired:

| identical | kept | retired |
|---|---|---|
| functions | the oldest, and, where it has no child, the child of the oldest of them that has one | the newer functions; a child no kept function points to dies when its last open activation's window closes (R38) |
| classes | the oldest, taking the others' places in the functions that use them | the newer classes |

What merging retires goes on the return with everything else retired this call (§5.7), so the machine deletes
the children on its ordinary pass.

## 5.6 Retire

> **R18 — Retire.** After the frame's rows are fitted, the functions re-centered and the seeds tried (R20),
> price every function's margin against the table as it now stands (D30), and retire every one that is strictly
> negative. Retiring is deletion from the table (D32). The function leaves that instant: it is bid no more, and
> the rows it fitted are written by the rest of the table. A class no function uses any longer is deleted with
> the last (R41). The retired functions go on the return (§5.7), and the machine deletes their children (§6.5).
> The neuron keeps no retired state.

Retiring comes after creating, so a short function and a longer one over the same ground are priced against
each other in the call the longer one is born, and the one that explains no row best leaves then.

## 5.7 Return

The call returns two lists: **the bids** of each of the frame's activations (D31, R21), and **everything
retired** this call, by deletion (R18) or by merging (§5.5). A bid is a function of the table: its id, its
relations and variables, and its child (D31). A function that
has never won a bid has no child, and its bid says so: if it is accepted the machine gives it a child, reused or
new (R16, R43), and points it there by the update child call (§4.1). A function retired is named by its id, and the machine
deletes its child, if it has one, when it is due (R18).

> **R21 — Bids of an activation.** An activation sends one bid (D31) per function of its neuron's table that
> fits its neighborhood. An activation no function fits sends nothing.

## 5.8 Cost of the call

What one call costs, with `H` rows in the history, `A` rows admitted this frame, `N` neighbors per row, `F`
functions in the table, `P` places per function, `K` members per class, `S` seeds tried, and `U` the uses of
a function, at most `H`:

| Step | Work | What it is |
|---|---|---|
| refresh history | `A · N²` | every relation of each admitted or evicted row (D47) |
| fit | `A · F · P` | each function against each admitted row (D54) |
| re-center | `F · U · N · (P + K)` | per function whose uses changed, each candidate place over its uses, four options (D27); then its fit re-read, `H · P` |
| create | `S · U · (N · (P + K) + F · P)` | per seed, the collapse over the rows that hold it, then its margin: a row's cost with and without it, over those rows (D33, D30) |
| merge | `(F + S) · P` | a canonical form per added or re-centered function, compared by lookup (D43) |
| retire | `F² · P · U` | per function, a row's cost with and without it over its uses; a row's cost is a greedy cover, `F · P` (D30, D28) |
| return | `A · F · P` | the fits of the frame's rows (D31) |

Retire and create dominate. Everything scales with `H`, `F`, `N` and `S`, and nothing with the run. Fit and
return are incremental in `A`; re-centering, create and retire read uses, which are `O(H)`. The margin is the
cost center: a row's cost is a greedy cover of the row by the whole table, and every margin asks for it twice
per row. The benefit is zero in a row a function does not fit, so only its uses count. What of this is cached,
and how, is the implementation's.

# Part III — Past and present: the machine

# 6. Contraction

> **D23 — Contraction.** The machine covers the frontier with children, each taken when
> the neurons it covers cost more than its bid costs to state, never counting its entry in the dictionary (R22, R24).
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
| activate children | Fire every child the level before bought, each at its bidder's coordinate (D53); the uncovered stand on the apex, and what the bids bound stays there. |
| process functions | Call every neuron with an activation at this level (§5).                                                     |
| elect bids        | Cover the frontier's activations that this level's bids name, by the greedy cover over the board.            |
| create children   | Give every accepted bid whose function has no child one: reused where an interneuron already stands for the same thing (D36, R43), created otherwise. It fires at the next level. |

Then, once the last level has run:

| Step              | Description                                                                                                   |
|-------------------|---------------------------------------------------------------------------------------------------------------|
| update children   | Call update child for every function whose bid won a child this frame. Then, for every interneuron marked changed, write its new definition to the interneuron index, and call update child for each of its parents, with the function of that parent that points at the child (§4.1, D36). |
| delete children   | After the frame's connect and infer calls (§7), delete every child that is due (R38).                        |

## 6.1 Process levels

**A level is explained as a set of calls with their arguments** (D37). Every accepted bid is a call: a
function's child, and beside it, bound on the apex, what stood in each of the function's variables. What stands
above is made of the calls and their arguments (§6.4): a function there over a child alone holds for every
argument the call was given, and one over the child and a bound activation holds for that one argument.

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

> **R22 — What a bid covers, and what it costs.** The neuron sends the bid, its function (D31), and nothing
> else. The machine holds the frame, so it reads that one object against what fired and derives both numbers.
>
> | | |
> |---|---|
> | the bid | its function, and its child or none (D31); the machine resolves where each node stands and what each variable binds |
> | coverage | the activations it covers that stand on the apex (D24), its bidder among them, each at what it costs there; and the positions of the activations it binds |
> | price | its record (D12, D22): its child and its position; and an absence for each neuron it names in those same frames that did not stand |
>
> The machine resolves the bid from the bidder's coordinate by D31's rule. Coverage less price is the savings
> over stating what it covers flat, D22's expression over the machine's population.
>
> **A position is covered once.** A bid that binds an activation whose position an earlier accepted bid covered
> is credited nothing for it (D10, D24).
>
> **A neuron that fired and the bid does not name belongs to neither side.** It stands in the file as itself if
> nothing covers it (D23), and the same way if this child is promoted, since the child does not name it — its own
> cost either way, so it cancels before the test begins, and charging it here would count it twice against the
> uncovered term of the same sum (§6.2).
>
> **Coverage changes the credit and never the price.** A neuron the bid names that fired and another bid
> already covers is credited to no one and charged nothing: it fired, so it was never among the neurons named
> and absent. A neuron the bid names that did not fire is charged an absence whether or not another neuron is
> right at that coordinate — another neuron's expansion being right there does not make this one's wrong name free. **What a
> pattern gets wrong about a frame is a fact about the two, and ownership does not move it.**
>
> **This is the neuron's arithmetic over the machine's population, and the number is not the neuron's.** The
> neuron offered the function on its own history (R20); the machine tallies on a
> board where earlier frames' credit stands (R23) and this frame's other bids contend (R24). The two numbers
> differ, and are meant to (D22).
>
> **This is a price for one bid, not for the symbol.** The entry in the dictionary is weighed by the one test
> (D30) and appears nowhere in this price and nowhere in the election.

**Contraction proposes nothing.** Every candidate comes from a neuron's own history, and the machine only
accepts or declines one — it never edits a bid, merges two, or invents a third. What it does do is *measure*
one: a bid arrives as a definition, and everything it is worth this frame the machine works out itself (R22).

> **D24 — Coverage set.** The machine keeps, for each activation an accepted bid was credited, two things:
>
> | | |
> |---|---|
> | credited to | the accepted bid that was last credited the activation, by covering it or by binding it (R22) |
> | covered | whether that bid covered the activation, so that it left the apex, or bound it, so that it stays with its position covered (D10) |
>
> **Credit is what blocks later payment, and coverage is what hides.** A covered activation is settled. It is in
> no later bid's coverage: a bid that names it afterward is neither paid for it nor charged a failure for it,
> since it fired. A bound activation stays on the apex at its neuron's cost and is a neighbor to every later
> bid. A later bid that covers it is credited its neuron, which is all that is left of it in the file. A later
> bid that binds it again is credited nothing, since its position is covered already (D10).
> So every bit of an activation is paid for once.
> **An activation is named by its full coordinate**, dimension and position together, so two activations of one
> neuron at two positions never contend. The coverage set spans one frame more than the longest reach in time
> among the neurons the machine holds — no bid reaches further back — and the box every other activation
> dimension gives, and ages out with it, and over the same span the machine keeps the action that ran at each
> frame in each action channel, for the shares a reward sends back (R33). **The machine holds nothing on the
> scale of the run.**
>
> The record is about **credit**: an activation is a fact that needs paying for exactly once, so a covered one is
> settled and never revisited (R23). It is not about naming: a child expands to everything its function names,
> credited or not (R24).

> **R23 — This frame's bids against the board as it stands.** Only what stands on the apex is in play: a bare
> activation at its cost, and a bound one at its neuron's cost. So no bit of an activation is paid for twice. **No
> earlier promotion is ever re-scored.** Within the electing frame an activation is covered once, by the first
> accepted bid that covers it (R24), and never moves.
>
> **Earlier bidders have priority.** A bid at `f` wins a neuron at `f − 2` before a better bid at `f + 1` can
> name it, because the earlier election settled it and nothing re-elects the past. That is the price of never
> revisiting a frame, and the design pays it.

**The file over one frame is the calls promoted plus the neurons no bid covered**: the price of every accepted bid,
and what each neuron no bid covered cost, the transcript half of `L` (§3.9) over the frames the election can see.
**The two terms do not overlap**: a neuron a promoted neuron fails to name is in the second and not the first,
which is why the price counts only the neighbors named and absent (R22). The dictionary half is D30's, and neither
test touches the other's sum.

**That sum is the objective; R24 is the procedure that serves it.** **Nothing anywhere forms a subset of bids
and scores it** — bids are taken one at a time, each measured against what the ones before it left, and the
election stops at the first that does not pay. **The election does not minimize the sum**: it is greedy, and
returns a good solution, not a proved minimum. **Every neuron a bid covers ends up credited to exactly one
bid**, which is what stops an activation being paid for twice.

> **R24 — The election is D28, run by the machine.** The functions are this frame's bids and what they cover is
> the **free set**: every activation on the frontier (R27), at its own full coordinate — frame and position — that
> some bid names and no earlier election has covered, a bound one at its neuron's cost (R23).
>
> Each bid is resolved on the board from its bidder's coordinate before the first round (D31).
> `price` is R22's. **Two bids that measure the same are told apart in this order**: a bid carrying a child
> before one carrying none; between two with children, the older child; between two with none, the older
> bidder; and between two activations of one neuron, the earlier coordinate, the earlier frame first and then
> the smaller position, the dimensions in the order the machine declares them (D1). Older is the smaller id
> (D2). Then D28 runs, and what it takes are the accepted bids.
>
> **The bound is structural.** A bid's price includes its child, a neuron, so a bid over a base bidder is
> taken only when it covers more than that: at least one neighbor beside the bidder. Each round takes one bid, so the rounds are
> never more than the bids.
>
> **A bid that never reached the top held nothing.** There is nothing to hand back and nothing to settle:
> an activation it named is either credited to a bid that did pay or stands as itself.
>
> **Ownership is a partition of the neurons the accepted bids name**, and that is the whole of the
> inhibition — no bid is ever edited or forbidden, and **overlap is legal and priced**: a bid that names an activation
> an earlier round covered gains nothing for it and pays nothing for it. One that covers a bound activation gains
> its neuron's cost, and one that binds it again gains nothing (D10, R22). **Held by an accepted bid** and
> **named by an accepted bid** are therefore the same set, so coverage, credit and the apex frontier (R27) are
> one question with one answer.
>
> **Outcome**: accepted bids are promoted, each **whole**, and a child expands to everything its function
> names, credited or not; the activations they cover leave the apex and what they bind stays on it (D24), and
> every active neuron no accepted bid covers stands as itself. **The election delivers
> nothing to anyone**: it writes the coverage set and stops. No neuron is told which of its bids were bought, what
> they were credited, or what they lost; a neuron's history is what it saw, and the board is the machine's.

## 6.3 Create children

Every accepted bid whose function has no child is given one. A bid that lost is given nothing.

> **R43 — Reuse or new.** The candidate for reuse is the interneuron that stands for what the accepted bid's
> child would, found through the index (D36), wherever it is: a neuron that bid this frame, one that fired
> frames ago and has stood for the same thing since, or one that has not fired within any open window. A stretch
> that comes round again therefore fires the neuron minted when it first came round, and not a new one above it.
>
> **Identical bids share one child.** Two bids are identical when their children would stand for one thing: each
> names exactly the activations the other names, its bidder among them, at the same price (D31), two functions
> stating one set of activations from two of its neurons. Every neuron the accepted bid covered was called this
> frame and bid every function that fit (R21), so such a function is among this election's bids. There are three
> cases, and each ends with one child.
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
> **Otherwise the machine creates one**: an id, its parent and an empty table (R16), its first activation at
> the bidder's coordinate (D53).
>
> **Nothing is sent back during the levels.** The function stays the neuron's own and keeps re-centering on its
> own history (D29); only the pointer is shared. A function that drifts from the definition of the interneuron
> it points at sets that definition when its bid is next accepted (D36). The neuron optimizes its history and
> the machine its window.
>
> **The machine keeps track of definitions.** For every accepted bid, the machine compares the bid's function
> with the definition of the interneuron it points at, in the interneuron index. Where the bid sets the
> definition (D36), the machine marks the interneuron changed and keeps the new definition until update children.

Once the frame's last level has run, the machine calls update child for every function whose bid won a child,
and for every parent of an interneuron marked changed (§4.1).

## 6.4 Activate children

A child fires at the level after the one its bid was accepted at, before that level is called. It fires at the
bidder's coordinate (D53), and what stood in its function's variables stays on the apex beside it, bound (D10).
Nothing sees a child until it fires (D5). A child's activation records the function whose bid fired it, since
it may have several (R43), and is expanded through that one (R28). Every activation no accepted bid covers
stands as itself.

> **R27 — The apex is a frontier.** It is every active neuron **no accepted bid covers** (D49) — the
> uncovered set, base neurons and interneurons alike — so a base neuron no pattern found worth naming stands in it beside a
> child many patterns deep. This is the frontier the file counts, **the one every neuron sees as its neighborhood**
> (D5), **the one that learns what ran** (D25), and **the one that votes** (§8.4): the uncovered set does all four,
> and coverage silences a neuron in every one of them at once, its one last connection aside (D10). A reward for a
> frame already written reaches its connection regardless (R33). Everything underneath the current frontier is
> recovered by expanding it.
>
> **Uncovered, not childless.** A neuron that offered a child and had its bid declined with nothing else
> covering it is still on the frontier, and stands in the file as itself.
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

> **R38 — A child dies when the last function pointing at it has retired and the last window over it closes.** A pattern the
> neuron retired (R18) names a child the machine still holds, unless no bid of it was ever accepted and there
> is nothing to delete (R16). The machine wired every pattern that promotes the child (R43), so it knows when
> the retired one was the last; while another remains the child lives and only the retired pattern goes. The
> neuron says nothing about when the child should go: it sees its own open activations and not the children
> promoted off them, while the machine sees both. The death frame is the frame in which the last open
> activation of the child, or of any neuron in its subtree, closes its window, and this frame when none is open.
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
> **The ledger holds the function, not a handle to it.** A child is stated by the patterns wired to it (D12),
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

The neuron chooses nothing in either call and is told nothing of what came of its inferences. Choosing among the
inferences of every activation is the machine's, and is Part V (§8).

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
| the apex | every activation standing on the apex this frame within the activation's reach in every dimension but time (R27, D25), and only if the activation is uncovered at this frame. One an accepted bid covered this frame is handed the neuron that covered it and nothing else; one covered earlier writes nothing more (D10, D25, R31). |
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
> what its bid covered and bound: the nodes its function names, the one whose bid fired the activation (§6.4),
> the owner's activation at the child's coordinate and each other node at its offset from it by the relations
> (D34) — offsets compose because each is a difference of activation coordinates (D2) — repeated down to base
> symbols. A fixed node expands to its neuron; a variable expands to its argument, the neuron bound there (D37).
> So `A` placed at `f`, whose function in `v`'s table is `u` at `−1` from `v`, expands to `v` at `f` and `u` at
> `f − 1`; and `P` placed at `f + 4`, whose function in `b`'s table is `C` at `−2` from `b`, `C`'s function in
> `c`'s table being `d` at `−1` from `c`, expands to `b` at `f + 4`, `c` at `f + 2` and `d` at `f + 1`.
> **A coarse offset expands to its rounded coordinate.** A neighbor named within the radius is placed where it
> fired. One named beyond the radius is placed at exactly the distance its offset names, whatever distance in
> the group it fired at, and several neighbors at one coarse offset (D6) are each placed there. **The rounding
> composes.** Each expansion down adds its own group's slack, so a neuron places the base symbols of its
> farthest neighbors to within the sum of the groups along the path: the longer its span, the coarser its far
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
> no argument places its most frequent member, the oldest on a tie. A function above that names a child alone
> expands to the call with its arguments left to be bound, and one that names the child and a bound activation
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
> like a bought one: it runs `process functions`, may be covered (§6.1), and every uncovered activation open
> at that frame connects to it (D25). While its
> program runs, what stands on the apex at each frame is the base action it placed there, or a lower pattern as
> it completes; the whole stands on the apex only in the frame it completes, and what the program earned reaches it through the span a
> reward names (R33).

## 8.3 Match

> **R49 — Match.** The machine reads one standing inference (R45) against the frames its function reaches
> back into, and proposes it where the function would pay there and strikes it where it would not. It runs
> every frame the inference stands, over the part of the function that has come to lie in the past. Nothing competes in it,
> nothing is credited and nothing is written: it is the one valuation (D22) read over its third set (§4.5).
>
> **What is read.** The inferred child's function, as it stands in the table (R28): its fixed nodes, each at
> an offset, and its variables, each at its places, placed back from the completion the inference names, one
> expansion down and no deeper. The bidder, and whatever is placed at the frame ahead or beyond, is what the
> inference proposes and is not judged. Against them stands the apex of each frame within the longest reach in
> time, as the machine keeps it (D24).
>
> **What each placed node is worth.**
>
> | Placed node | Worth |
> |---|---|
> | a neighbor standing at its offset | credits what it cost |
> | a neighbor standing within reach at another offset | credits what it cost, and charges an offset (D13) |
> | a neighbor standing nowhere within reach | charges an absence (D13) |
> | a variable | bound from what stands at its places that have happened (D44), crediting their positions; a place holding nothing it could take charges an absence |
>
> An offset is read as D6 reads it: exact within the radius, and beyond it as its whole group, composed down the
> expansion as the placement is, so a neighbor named at a coarse offset stands at its offset anywhere in the
> group. A neighbor found at a different distance is a correction that says where, and it costs what an offset
> costs, near or far: what makes a near miss cheap is that within the group it costs nothing.
>
> **The test.** The inference is proposed when what the placed nodes credit exceeds what they charge, absences
> included, and struck otherwise. That is D22's test with the bidder on neither side, since the bidder
> is ahead.
>
> **What it decides.** Proposed, the inference's placements ahead contend for their positions (R46, R47).
> Struck, it leaves the standing inferences (R45). An inference that was proposed and stops matching as its
> placements come due is struck then, which is the whole of how a plan ends before its span does.
>
> **Where it stands among the three.** A neuron reads a function against its own rows and prices its table
> (D30). The election reads bids against the board and credits them (R24). The match reads one
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
| Alternation | [algorithm-xy.md](algorithm-xy.md) | A stream at radius 4. A neuron that keeps seeing a different pair of letters holds one function with a parameter tying the letters two frames apart, twice over, learned from the pairs that go by once: its child is the alternation, one neuron for every such pair, and the letters bound beside it say which pair. A pair that comes often is named by its own letters' patterns first, and stands in a class instead. |
| Addition | [algorithm-addition.md](algorithm-addition.md) | A request with each column, a neuron for each case that can be asked, a function per case, and a carry the world shows while it teaches and the machine infers afterwards. |
| Copy | [algorithm-copy.md](algorithm-copy.md) | Shown a digit and asked, the machine says it: one function in the answer's table with a parameter tying the digit shown to the digit said, bound from what was shown, so a digit never said before is said the first time; once a class binds the shown digit the same parameter reads it there. |
| Equal | [algorithm-equal.md](algorithm-equal.md) | Shown four digits in one frame and asked, the machine says the third where the first two are the same and the fourth where they differ: equality is a parameter over two variables of the request, the answer is a copy by a parameter over a variable of the request and the variable of the answer, bound from what was shown, and the request answers by a vote the one voter that knows the situation decides. |
| Hit | [algorithm-hit.md](algorithm-hit.md) | Offsets from where the world reports something approaching as the reference frame, a parameter that takes whatever comes, bound beside the call, and a thing never seen answered by the lesson of the call neuron, which stands for anything coming that way. |
| Loop | [algorithm-loop.md](algorithm-loop.md) | Shown a cue, the machine taps in each of the next three frames and then stops. There is no counter: each tap's table names what stood a frame back, so the states of the loop are three children each built over the one before, each connected to the one after it and the last to the event the world shows at the end. Stopping is a thing that follows and not an absence, since speaking falls through the cover; taught to stop on a cue instead of a count, the same machine is a while loop. |
| MNIST | [algorithm-mnist.md](algorithm-mnist.md) | An image channel with two spatial dimensions and a binary pixel dimension, 256 buckets or three channels for color later; patches at the base, odd shapes closed on the frontier, a digit action taught by demonstration. |
| Stocks | [algorithm-stocks.md](algorithm-stocks.md) | A price channel per instrument with own and disown as its actions, under a policy where an action is no one's neighbor and only events and patterns connect, to actions alone; patterns over frames; a world that runs the right action wherever the machine has none; and a reward scoped to the channel and the frame, each frame. |
| Text | [algorithm-text.md](algorithm-text.md) | A stream of letters at reach 1: at the base a class over the letters that stand before each letter, the letter bound beside the pair's call neuron; words of any length from children and letters on the frontier, a frequent doubled letter as a function in its own letter's table. |

# 10. Further reading

Each document is written in the terms of this specification and references its keys.

| Document | What it holds |
|---|---|
| [algorithm-remarks.md](algorithm-remarks.md) | The theorems, and the commentary on why the design is shaped this way, keyed by the same D and R numbers. |
| [algorithm-evaluation.md](algorithm-evaluation.md) | The falsifiable claim, the risks with what would be done about each, and the open questions. |
| [algorithm-implementation.md](algorithm-implementation.md) | The implementation plan: the per-neuron state and calls, the build stages, and the deltas against the current code. |
| [hippocampus.md](hippocampus.md) | An addendum: the second organ, running beside this machine over the same neurons, with rules keyed **H**. |