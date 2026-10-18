# Hit whatever comes at you, as a machine that could be built

A worked case for [algorithm.md](algorithm.md): a machine that hits whatever is coming at it, wherever the two
of them are, written out neuron by neuron. Nothing here is normative, and the machine is shown running: its
tables and connections are as listed, and its histories already hold moments like this one. It shows where the
design does its reference-frame calculation: the machine subtracts coordinates when it assembles a neighborhood
(D6), and a pattern is written in the offsets between its own nodes (D34), so everything that is learned or
acted on is written in differences, and a thing's course is written without saying where it was. And it shows
what answers for a thing never seen before: the situation written without the thing.

---

# 1. The environment

**In.** A field, laid out over two activation dimensions and time. What is in it is reported as events: `ball`
where a ball is, `fist` where a fist is. One more event is the world's own: `approaching` fires at the body's
position in every frame in which something is coming at the body, whatever it is.

**Out.** One action dimension with three base actions, none with an argument: `hit`, which strikes what is
straight ahead of the body, `turn-left` and `turn-right`.

# 2. What stands under them

The field declares a radius of `4` in each of its two spatial dimensions and `1` in time (D1), so `approaching`,
`ball` and `fist`, base events all, see four positions in every direction and one frame back (D4), and within that
reach they are each other's neighbors on the frontier (D5). Within the radius every distance is exact (D6); the
distances this case uses are `2` and `4`.

# 3. The neurons

| Kind | Neurons |
|---|---|
| events | `approaching`, `ball`, `fist`, and whatever else the field holds |
| parameters | `T`, `T′` and `T″`, in `approaching`'s table: two positions a frame apart that hold one and the same thing, one per course (D44) |
| the functions over them | `coming`, `coming-left` and `coming-right`, in `approaching`'s table, each naming `approaching` and one parameter beside it; their children `COMING`, `COMING-LEFT` and `COMING-RIGHT` (D48) |
| value neurons | `T:ball`, `T:fist` and so on: one per parameter and thing it has held (D45) |
| base actions | `hit`, `turn-left`, `turn-right` |
| the functions of the actions | in the table of each base action, the action and what stood a frame before it: a function naming a child or a value neuron that comes often, a class for the rest |

# 4. The parameters, and what each thing learns

Each parameter is two positions, the same thing at both, whatever it is (D38): one thing, two positions nearer
than it was a frame ago. A parameter is written in the offset between its positions (D44), so it says how the
thing moved and not where it was. The function that names `approaching` beside it says where, since
`approaching` fires at the body (D38).

| Parameter | where `coming` names it, from the body | its one offset | what its value neurons learn |
|---|---|---|---|
| `T`, coming straight on | `(0, +2)` now | `(0, +2)` further out, one frame ago | `hit` |
| `T′`, from the left | `(−2, 0)` now | `(−2, 0)` further out, one frame ago | `turn-left` |
| `T″`, from the right | `(+2, 0)` now | `(+2, 0)` further out, one frame ago | `turn-right` |

When `T` holds two positions ahead of the body, `coming` holds: a function naming `approaching` and `T` at
`(0, +2)` from it (D38). `approaching` sends it as one bid, with `T` and the thing it held (D31). The bid
covers `approaching` and both activations of the thing. `COMING` fires in the frame `approaching` fires, at the
center of what the bid covered, `(0, +2)` from the body, between the body and where the thing was (D53), and
`T`'s value neuron for that thing fires over `T`'s two positions, `(0, +3)` from the body (D45). A value neuron
is one thing on one course, `T:ball` a ball coming straight on, and each holds a lesson of its own: `T:ball`
connects to what follows it, `hit`, at the offset from where it stood to the body a frame later, with what the
hit earned (R31). `COMING` is anything coming straight on, and holds that lesson for all of them.

The function in `hit`'s table is the same thing seen from the action: it names `hit` and what stood a frame
before it, `COMING` two positions ahead and the value neuron three ahead (D5). Once it exists its bid covers
both in the frame `hit` runs, and in that frame each connects to its child, with that frame's reward (D10).
Inferred, that child expands to its bidder, `hit`, which is output (R28, R30).

`coming` pays because `T` has two positions and its member is paid once: `approaching` and two activations of
the thing are written as `approaching`, which of its patterns, and one member, a choice among the things `T`
holds.

**How `T` came to be.** Whatever came at the body stood in `approaching`'s residual at `(0, +2)` and, a frame
earlier, at `(0, +4)`: two offsets that hold one and the same neuron in each use, whatever it was (D47). The
parameter's collapse made them one parameter, `T`, the offset between them (D27), and `approaching` standing two
positions behind `T`'s first position in every use made the function `coming` over the two (D33). No function
names a thing at those coordinates: once `T` holds them they are `T`'s, and a thing that comes often is `T`'s
value neuron for it (D27).

**The habit.** Before `coming` existed, `approaching` stood uncovered while things came and `hit` ran, so it
holds a connection of its own: after `approaching`, `hit`, with what those hits earned. Once `coming` holds,
`approaching` is covered in the frame it fires and writes nothing more (D10). What it wrote stands, since
nothing leaves a connection (R31).

# 5. One moment, worked

The body is at `(10, 10)`. A ball is at `(10, 12)`, and a frame ago it was at `(10, 14)`. `approaching` fires
at `(10, 10)`.

1. **The machine subtracts.** Assembling the neighborhood of the `approaching` activation, it takes the
   difference of coordinates and buckets each component (D6): the ball now is at `(0, +2)`, the ball a frame
   ago at `(0, +4)`.
2. **The parameter holds, and the function with it.** `T`'s two positions are two apart and a frame apart, and
   the same member stands at both; `approaching` stands two behind the first. `approaching` sends `coming` with
   `T` holding `ball` (D31). The bid is accepted, covering `approaching` and both activations of the ball, and
   one level up `COMING` fires at `(10, 12)`, the center of the body and the ball's two positions, and `T:ball`
   at `(10, 13)` (§7.4).
3. **The value neuron speaks.** Its connection names the child of the function in `hit`'s table, one frame on
   and three positions back, where the body is (D25). That is the entire record, and `COMING` holds the same
   one from two positions back.
4. **Next frame the child is expanded and `hit` runs** (R28, R30). Its bid covers `COMING` and `T:ball`, which
   connect to it once more, with what the hit earned (D10).

# 6. Wherever the two of them are

Put the body at `(50, 3)` and a ball at `(50, 5)` that was at `(50, 7)`. The subtraction gives the same two
offsets, the same parameter holds, the same value neuron fires, and the same connection infers the child that
hits. No coordinate ever reached the pattern (D11), so where they are costs nothing.

# 7. A thing never seen

A rock comes from ahead. It stands at both of `T`'s positions, and for a parameter that is the whole test, so it
is `T`'s value on the spot and a member from then on (D27), and the machine gives it a value neuron, `T:rock`
(R44). `T:rock` has never fired and holds no connection, so it has nothing to infer. `COMING` has: it fired for
every ball and fist that came straight on, and its lesson says `hit`. The rock is hit, the hit is rewarded, and
`T:rock`, open and uncovered a frame later, connects to `hit` with that reward. From then on a rock is hit on
its own lesson. Before `coming` existed, `approaching`'s own habit under the cover would have answered the same
way, since what a coverer with nothing to say covers speaks in its place (R48).

So what generalizes is the situation written without the thing: the function's child, which says how something
came and leaves what it was to the parameter, and it answers exactly once per new thing.

# 8. Wherever it is relative to the body

That costs a parameter and a function per direction. A ball coming from the left gives `(−2, 0)` now and
`(−4, 0)` a frame ago, an offset of `(−2, 0)` between them a frame apart, which is `T′`, and `coming-left`
names `approaching` two positions to its right; `COMING-LEFT` and `T′:ball` have learned the child that turns
left. After the turn the ball is ahead, `COMING` and `T:ball` fire, and `hit` follows. The frame of reference
is never transformed: the body turns, the field is reported again, and the machine subtracts again. The radius
is what keeps the directions and distances to a handful: four positions each way, exact, and nothing beyond
(D1, D6).

A thing never seen that comes from the side is answered by `COMING-LEFT`, which has seen only turns pay. Where
that is wrong the reward says so, the new value neuron's estimate for it turns negative, and the walk tries the
next action (R37).

# 9. The same thing as a reference frame

| in a reference-frame account | in the machine |
|---|---|
| the frame's origin | none in the pattern: a relation is an offset between two nodes (D34), and the child is placed over what it covered (D53) |
| a location in the frame | an offset from another node (D6, D34) |
| a feature at a location | a node in a relation (D16, D34) |
| the same object at any place in the world | one pattern, since no coordinate is part of a neuron (D11) and no pattern says where it stood (D13) |
| any object at that location | a class a function names at the offset (D41) |
| the same object at two locations | one parameter, the offset between them (D44) |
| what to do about it | the value neuron's connection to a function's child, which names no location at all |
