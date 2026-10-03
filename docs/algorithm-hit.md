# Hit whatever comes at you, as a machine that could be built

A worked case for [algorithm.md](algorithm.md): a machine that hits whatever is coming at it, wherever the two
of them are, written out neuron by neuron. Nothing here is normative, and the machine is shown running: its
tables and connections are as listed, and its histories already hold moments like this one. It shows where the
design does its reference-frame calculation, which is in one place and only one: the machine subtracts
coordinates when it assembles a neighborhood (D6), and everything that is learned or acted on afterward is
written in the differences. And it shows what answers for a thing never seen before: the habit of the one event
the world reports in every case.

---

# 1. The environment

**In.** A field, laid out over two activation dimensions and time. What is in it is reported as events: `ball`
where a ball is, `fist` where a fist is. One more event is the world's own: `approaching` fires at the body's
position in every frame in which something is coming at the body, whatever it is.

**Out.** One action dimension with three base actions, none with an argument: `hit`, which strikes what is
straight ahead of the body, `turn-left` and `turn-right`.

# 2. What stands under them

What stands under `approaching`, `ball` and `fist` is stated, not drawn: each is taken to reach four steps (D4),
and within that reach they are each other's neighbors on the frontier (D5). Offsets are bucketed by powers of
two (D6), so the distances that matter here are `2` and `4`.

# 3. The neurons

| Kind | Neurons |
|---|---|
| events | `approaching`, `ball`, `fist`, and whatever else the field holds |
| parameters | `T`, `T′` and `T″`, in `approaching`'s table: variables of two positions, the same thing at both, one per direction (D41) |
| value neurons | `T:ball`, `T:fist` and so on: one per parameter and thing it has held (D45) |
| base actions | `hit`, `turn-left`, `turn-right` |
| the steps | in the table of each base action, what stood a frame before it: a function for a value neuron that comes often, a class for the rest |

# 4. The parameters, and what each thing learns

Every parameter is in `approaching`'s table, so every offset is measured from the body. Each has two positions,
which means the same thing at both, whatever it is (D38): one thing, nearer than it was a frame ago.

| Parameter | its positions | what its value neurons learn |
|---|---|---|
| `T`, from ahead | `(0, +2)` now and `(0, +4)` one frame ago | `hit` |
| `T′`, from the left | `(−2, 0)` now and `(−4, 0)` one frame ago | `turn-left` |
| `T″`, from the right | `(+2, 0)` now and `(+4, 0)` one frame ago | `turn-right` |

When `T` holds, `approaching` sends it as a bid of its own (D31). No function names it: a function over one
variable and nothing else would write exactly what the variable writes alone, so none is built (D30). The bid
covers `approaching` and both activations of the thing, and `T`'s value neuron for that thing fires at
`approaching`'s coordinate. A value neuron is one thing coming from one direction, `T:ball` a ball from ahead,
and each holds a lesson of its own: `T:ball` connects to what follows it, `hit`, with what the hit earned (R31).

A step is the same seen from the action: in `hit`'s table, the value neuron that stood a frame before (D5).
Once it exists its bid covers that value neuron in the frame `hit` runs, and in that frame the value neuron
connects to the step, with that frame's reward (D10). Inferred, the step expands to its bidder, `hit`, which is
output (R28, R30).

`T` pays because it has two positions and its member is paid once: `approaching` and two activations of the
thing are written as `approaching`, which of its patterns, and one member, a choice among the things `T` holds.

**How `T` came to be.** Whatever came at the body stood in `approaching`'s residual at `(0, +2)` and, a frame
earlier, at `(0, +4)`: a pair of offsets that hold one and the same neuron in each neighborhood, whatever it
was (D47). The variable's collapse made them one parameter, `T` (D27). A thing that comes often enough can also
earn a function of its own, with that thing fixed at both offsets, once the choice it stops writing pays for
the line (D30); when the thing turns rare and its function retires, it is written through `T` again.

**The habit.** Before `T` existed, `approaching` stood uncovered while things came and `hit` ran, so it holds a
connection of its own: after `approaching`, `hit`, with what those hits earned. Once `T` holds, `approaching`
is covered in the frame it fires and writes nothing more (D10). What it wrote stands, since nothing leaves a
connection (R31).

# 5. One moment, worked

The body is at `(10, 10)`. A ball is at `(10, 12)`, and a frame ago it was at `(10, 14)`. `approaching` fires
at `(10, 10)`.

1. **The machine subtracts.** Assembling the neighborhood of the `approaching` activation, it takes the
   difference of coordinates and buckets each component (D6): the ball now is at `(0, +2)`, the ball a frame
   ago at `(0, +4)`.
2. **The parameter holds.** `T` has exactly those two positions, and the same member stands at both.
   `approaching` sends `T` with `ball` (D31). The bid is accepted, covering `approaching` and both activations
   of the ball, and one level up, at the body's coordinate, `T:ball` fires (§7.4).
3. **The value neuron speaks.** Its connection names the step, one frame on (D25). That is the entire record:
   no position is in it, and none is needed, because `hit` strikes what is ahead of the body and `T` only holds
   when something is.
4. **Next frame the step is expanded and `hit` runs** (R28, R30). The step's bid covers `T:ball`, which connects
   to the step once more, with what the hit earned (D10).

# 6. Wherever the two of them are

Put the body at `(50, 3)` and a ball at `(50, 5)` that was at `(50, 7)`. The subtraction gives the same two
offsets, the same parameter holds, the same value neuron fires, and the same connection infers the step that
hits. No coordinate ever reached the pattern (D11), so where they are costs nothing.

# 7. A thing never seen

A rock comes from ahead. It stands at both of `T`'s positions, which is enough to pay for its entry, so it
joins `T` on the spot (D27), and the machine gives it a value neuron, `T:rock` (R44). `T:rock` has never fired
and holds no connection, so it has nothing to infer. Where a neuron on the apex has nothing to say, what it
covers speaks in its place (R36): `approaching` is under it, and `approaching`'s habit says `hit`. The rock is
hit, the hit is rewarded, and `T:rock`, open and uncovered a frame later, connects to `hit` with that reward.
From then on a rock is hit on its own lesson.

So what generalizes is not a neuron the machine built. It is the event the world reports in every case, and it
answers exactly once per new thing.

# 8. Wherever it is relative to the body

That costs a parameter per direction. A ball coming from the left gives `(−2, 0)` and `(−4, 0)`, which is `T′`,
and `T′:ball` has learned the step that turns left. After the turn the ball is ahead, `T:ball` fires, and `hit`
follows. The frame of reference is never transformed: the body turns, the field is reported again, and the
machine subtracts again. D6's bucketing is what keeps the directions and distances to a handful.

A thing never seen that comes from the side is answered by the same habit, and `approaching` has seen hits and
turns follow it alike, so it says whichever has paid best. Where that is wrong the reward says so, the new value
neuron's estimate for it turns negative, and the walk tries the next action (R37).

# 9. The same thing as a reference frame

| in a reference-frame account | in the machine |
|---|---|
| the frame's origin | the activation whose neighborhood is being read, here `approaching` |
| a location in the frame | an offset from that activation (D6) |
| a feature at a location | a neighbor, `(neuron, offset)` |
| the same object at any place in the world | one pattern, since no coordinate is part of a neuron (D11) |
| any object at that location | a class at the offset (D41) |
| the same object at two locations | one parameter over two offsets (D41) |
| what to do about it | the value neuron's connection to a step, which names no location at all |
