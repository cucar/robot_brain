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
| class neuron | `Thing`, whose members are `ball`, `fist` and the rest, but not `me` (D41) |
| base actions | `hit`, `turn-left`, `turn-right` |
| patterns | three, in the table of `me`, each with a child |

# 4. The situations, and what each returns

Every pattern is in `me`'s table, so every offset is measured from the body. Each names `Thing` twice with one role, `Thing¹`, which
means the same member at both offsets (D38): one thing, nearer than it was a frame ago.

| Child | names | returns |
|---|---|---|
| `incoming-ahead` | `(Thing¹, (0, +2), now)`, `(Thing¹, (0, +4), one frame ago)` | `hit` |
| `incoming-left` | `(Thing¹, (−2, 0), now)`, `(Thing¹, (−4, 0), one frame ago)` | `turn-left` |
| `incoming-right` | `(Thing¹, (+2, 0), now)`, `(Thing¹, (+4, 0), one frame ago)` | `turn-right` |

Each pattern pays for its line because `Thing` is named twice and bound once: the body and two activations of
the thing are written as the child and one binding.

# 5. One moment, worked

The body is at `(10, 10)`. A ball is at `(10, 12)`, and a frame ago it was at `(10, 14)`.

1. **The machine subtracts.** Assembling the neighborhood of the `me` activation, it takes the difference of
   coordinates and buckets each component (D6): the ball now is at `(0, +2)`, the ball a frame ago at `(0, +4)`.
   Beside each `ball` activation stands a `Thing` activation bound to `ball`, which the `ball` reported (§6.6).
2. **The pattern fits.** `incoming-ahead` names `Thing` at exactly those two offsets, and the same member stands
   at both. Its child fires at the body's coordinate, carrying `Thing = ball`.
3. **The child speaks.** Its connection is `(hit, one frame on)` (D25). That is the entire record: no position is
   in it, and none is needed, because `hit` strikes what is ahead of the body and the situation only fires when
   something is.
4. **Next frame `hit` runs.**

# 6. Wherever the two of them are

Put the body at `(50, 3)` and a fist at `(50, 5)` that was at `(50, 7)`. The subtraction gives the same two
offsets, `Thing` is bound to `fist` instead, the same pattern fits, the same child fires, and the same
connection returns `hit`. No coordinate ever reached the pattern (D11), so where they are costs nothing, and
the class neuron means what they are costs nothing either.

# 7. Wherever it is relative to the body

That does cost something, and what it costs is a situation per direction. A ball coming from the left gives
`(−2, 0)` and `(−4, 0)`, which is `incoming-left`, and its connection returns `turn-left`. After the turn the
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
| any object at that location | a class neuron named at the offset (D41) |
| the same object at two locations | the same class neuron and role named at two offsets |
| what to do about it | the child's action connection, which names no location at all |
