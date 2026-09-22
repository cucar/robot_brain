# Hit whatever comes at you, as a machine that could be built

A worked case for [algorithm.md](algorithm.md): a machine that hits whatever is coming at it, wherever the two
of them are, written out neuron by neuron. Nothing here is normative, and the machine is shown running: its
tables and connections are as listed, and its histories already hold moments like this one. It shows where the
design does its reference-frame calculation, which is in one place and only one: the machine subtracts
coordinates when it assembles a neighborhood (D6), and everything that is learned or acted on afterward is
written in the differences.

---

# 1. The environment

**In.** A field, laid out over two activation dimensions and time. What is in it is reported as events. The body
is one of those events: `me` fires wherever the body is.

**Out.** One action dimension with three base actions, none with an argument: `hit`, which strikes what is
straight ahead of the body, `turn-left` and `turn-right`.

# 2. The level

`me`, `ball` and `fist` are not base neurons. Each is the top of an ordinary stack of patterns below it, a body
or a ball as the lower levels have put it together out of edges and patches, and none of those lower patterns
names a class neuron. They stand at a level whose reach spans four steps (D4), and at that level they are each
other's neighbors (D5). Offsets are bucketed by powers of two (D6), so the distances that matter here are `2`
and `4`.

# 3. The neurons

| Kind | Neurons |
|---|---|
| events at the level | `me`, `ball`, `fist`, and whatever else the field holds |
| class neuron | `Thing`, a variable that holds whatever is coming (D41) |
| base actions | `hit`, `turn-left`, `turn-right` |
| patterns | three situations in the table of `me`, each with a child; three steps, one in the table of each base action, each naming the situation's neighbors a frame before and the action beside it |

# 4. The situations, and what each returns

Every pattern is in `me`'s table, so every offset is measured from the body. Each names `Thing` at two offsets,
which means the same value at both, whatever it is (D38): one thing, nearer than it was a frame ago.

| Child | names | connects to, a frame on |
|---|---|---|
| `incoming-ahead` | `(Thing, (0, +2), now)`, `(Thing, (0, +4), one frame ago)` | the step `hit`, in `hit`'s table |
| `incoming-left` | `(Thing, (−2, 0), now)`, `(Thing, (−4, 0), one frame ago)` | the step `turn-left` |
| `incoming-right` | `(Thing, (+2, 0), now)`, `(Thing, (+4, 0), one frame ago)` | the step `turn-right` |

A step is the same pattern seen from the action: in `hit`'s table, `me` and `Thing` at those offsets a frame
before, and `hit` beside it (D5). Matched, it says the body hit something incoming; inferred, it hits (R30).

Each pattern pays for its line because `Thing` is named at two offsets and stands once: the body and two
activations of the thing are written as the child and one variable.

**How `Thing` came to be.** Before it, `me` held a pattern per thing that had come at it often: a ball ahead, a
fist ahead. The merge (D42) paired the two, which differ at the same two offsets and hold one neuron at both in
each, and generalized them into `incoming-ahead` with `Thing` at those offsets. The pair paid nothing on the
balls and fists, which their own patterns already wrote as one symbol; it paid on everything else that had
ever come at the body once or twice and been written flat. The specific patterns stay while balls and fists
are frequent, and `incoming-ahead` serves the rest.

# 5. One moment, worked

The body is at `(10, 10)`. A ball is at `(10, 12)`, and a frame ago it was at `(10, 14)`.

1. **The machine subtracts.** Assembling the neighborhood of the `me` activation, it takes the difference of
   coordinates and buckets each component (D6): the ball now is at `(0, +2)`, the ball a frame ago at `(0, +4)`.
2. **The pattern fits.** `incoming-ahead` has its slots at exactly those two offsets, and the same neuron stands
   at both. `me` bids the child, with a class bid naming the ball it was fit by (D31). The bid is accepted, and
   one level up the child fires at the body's coordinate and a `Thing` variable fires at the ball's, holding
   `ball` (§7.4).
3. **The child speaks.** Its connection names the step, one frame on (D25). That is the entire record: no
   position is in it, and none is needed, because `hit` strikes what is ahead of the body and the situation only
   fires when something is.
4. **Next frame the step is expanded and `hit` runs** (R30).

# 6. Wherever the two of them are

Put the body at `(50, 3)` and a fist at `(50, 5)` that was at `(50, 7)`. The subtraction gives the same two
offsets, `Thing` holds `fist` instead, the same pattern fits, the same child fires, and the same
connection infers the step that hits. No coordinate ever reached the pattern (D11), so where they are costs nothing, and
the slot means what they are costs nothing either: it takes whatever stands there, seen before or not.

# 7. Wherever it is relative to the body

That does cost something, and what it costs is a situation per direction. A ball coming from the left gives
`(−2, 0)` and `(−4, 0)`, which is `incoming-left`, and its connection infers the step that turns left. After the turn the
ball is ahead, `incoming-ahead` fires, and `hit` follows. The frame of reference is never transformed: the body
turns, the field is reported again, and the machine subtracts again. D6's bucketing is what keeps the directions
and distances to a handful.

# 8. The same thing as a reference frame

| in a reference-frame account | in the machine |
|---|---|
| the frame's origin | the activation whose neighborhood is being read, here `me` |
| a location in the frame | an offset from that activation (D6) |
| a feature at a location | a neighbor, `(neuron, offset)` |
| the same object at any place in the world | one pattern, since no coordinate is part of a neuron (D11) |
| any object at that location | a class neuron named at the offset (D38) |
| the same object at two locations | the same class neuron named at two offsets |
| what to do about it | the child's connection to a step, which names no location at all |
