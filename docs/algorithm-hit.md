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
names a class. They stand at a level whose reach spans four steps (D4), and at that level they are each
other's neighbors (D5). Offsets are bucketed by powers of two (D6), so the distances that matter here are `2`
and `4`.

# 3. The neurons

| Kind | Neurons |
|---|---|
| events at the level | `me`, `ball`, `fist`, and whatever else the field holds |
| parameters | `T`, `T′` and `T″`, in `me`'s table: the same thing at two offsets, one per direction (D44) |
| base actions | `hit`, `turn-left`, `turn-right` |
| the situations | three constants one level above `me`, each over `me` and one parameter's child, each with a child |
| the steps | three, one in the table of each base action, naming the situation's child a frame before and the action beside it |

# 4. The situations, and what each infers

Every parameter is in `me`'s table, so every offset is measured from the body. Each spans two offsets, which means
the same thing at both, whatever it is (D38): one thing, nearer than it was a frame ago. When one holds, `me`'s
bid for it covers the two cells, and the parameter's child and value neuron fire at the body's coordinate one
level up, beside `me` itself, which stands uncovered. A constant there over `me` and the parameter's child is the
situation:

| Situation, over `me` and | the parameter's offsets | connects to, a frame on |
|---|---|---|
| `incoming-ahead`: `T`'s child | `(0, +2)` now, `(0, +4)` one frame ago | the step `hit`, in `hit`'s table |
| `incoming-left`: `T′`'s child | `(−2, 0)` now, `(−4, 0)` one frame ago | the step `turn-left` |
| `incoming-right`: `T″`'s child | `(+2, 0)` now, `(+4, 0)` one frame ago | the step `turn-right` |

A step is the same seen from the action: in `hit`'s table, the situation's child a frame before and `hit` beside
it (D5). Matched, it says the body hit something incoming; inferred, it hits (R30).

`T` pays for its line because it spans two offsets and its value is paid once: two activations of the thing are
written as which pattern and one value, a choice among the things `T` has passed. The situation pays because `me`
and `T`'s child recur together, at one coordinate, on every approach.

**How `T` came to be.** Balls and fists came at the body often and got constants of their own. Everything else
that ever came at it, once or twice each, stood in `me`'s residual at `(0, +2)` and, a frame earlier, at
`(0, +4)`: two offsets filled most of the time by no one neuron, and by one and the same neuron in each
neighborhood. The parameter's collapse (D27) made them one parameter, `T`: one value, paid once, covering two
cells, and its child is what the situation is built on. When balls become rare and their constant retires, balls
are written through `T` like everything else, with nothing to join.

# 5. One moment, worked

The body is at `(10, 10)`. A ball is at `(10, 12)`, and a frame ago it was at `(10, 14)`.

1. **The machine subtracts.** Assembling the neighborhood of the `me` activation, it takes the difference of
   coordinates and buckets each component (D6): the ball now is at `(0, +2)`, the ball a frame ago at `(0, +4)`.
2. **The parameter fits.** `T` spans exactly those two offsets, and the same neuron stands at both. `me` bids
   it, carrying `T = ball` (D31). The bid is accepted, and one level up, at the body's coordinate, `T`'s child and
   its value neuron for `ball` fire, beside `me` (§7.4).
3. **The situation fits.** `me` and `T`'s child stand together at one coordinate, which is `incoming-ahead`'s
   constant; its bid is accepted and its child fires one level higher, at the body's coordinate.
4. **The child speaks.** Its connection names the step, one frame on (D25). That is the entire record: no
   position is in it, and none is needed, because `hit` strikes what is ahead of the body and the situation only
   fires when something is.
5. **Next frame the step is expanded and `hit` runs** (R30).

# 6. Wherever the two of them are

Put the body at `(50, 3)` and a fist at `(50, 5)` that was at `(50, 7)`. The subtraction gives the same two
offsets, `T` holds `fist` instead, the same parameter and the same situation fit, the same child fires, and the
same connection infers the step that hits. No coordinate ever reached the pattern (D11), so where they are costs nothing, and
the parameter means what they are costs nothing either: it takes whatever stands at both, seen before or not.

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
| any object at that location | a class at the offset (D38) |
| the same object at two locations | one parameter over two offsets (D44) |
| what to do about it | the situation's connection to a step, which names no location at all |
