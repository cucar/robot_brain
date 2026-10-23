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
| the functions | `coming`, `coming-left` and `coming-right`, in `approaching`'s table, each naming `approaching` and a parameter beside it, `T`, `T′` or `T″`: two places a frame apart that hold one and the same thing, one per course (D44); their children `COMING`, `COMING-LEFT` and `COMING-RIGHT` (D48) |
| base actions | `hit`, `turn-left`, `turn-right` |
| the functions of the actions | in the table of each base action, the action and what stood a frame before it: a function naming a child that comes often, with a class for the rest |

# 4. The parameters, and what learns

Each variable ties two places, the same thing at both, whatever it is (D38): one thing, two positions nearer than
it was a frame ago. A parameter is written in the offset between its places (D44), so it says how the
thing moved and not where it was. The function it belongs to names `approaching`, which fires at the body, and
that says where (D38).

| Parameter | where `coming` puts it, from the body | its one offset |
|---|---|---|
| `T`, coming straight on | `(0, +2)` now | `(0, +2)` further out, one frame ago |
| `T′`, from the left | `(−2, 0)` now | `(−2, 0)` further out, one frame ago |
| `T″`, from the right | `(+2, 0)` now | `(+2, 0)` further out, one frame ago |

When `T` holds two positions ahead of the body, `coming` holds: a function naming `approaching` and `T` at
`(0, +2)` from it (D38). `approaching` sends it as one bid, with the thing in `T` (D31). The bid covers
`approaching` and binds both activations of the thing. `COMING` fires in the frame `approaching` fires, at the
body, its bidder's coordinate (D53), and the thing stays where it was, bound (D10). `COMING` is anything coming
straight on, and it holds the lesson: it connects to what follows it, `hit`, a frame later at the body, with
what the hit earned (R31). The thing itself, a `ball` or a `fist`, connects too, but as the base neuron it is,
pooled over every place a ball has ever stood (D11); what it learns is a ball's habit, not a ball coming
straight on. A lesson per thing-on-a-course is one level up: a function over `COMING` and the bound `ball`,
whose child holds exactly that. It forms only once that pairing has stood twice in `COMING`'s history (R14),
so the first ball is answered by `COMING`, not by a lesson of its own.

The function in `hit`'s table is the same thing seen from the action: it names `hit` and what stood a frame
before it, `COMING` at the body (D5). Once it exists its bid covers `COMING` in the frame `hit` runs, and in
that frame `COMING` connects to its child, with that frame's reward (D10). Inferred, that child expands to its
bidder, `hit`, which is output (R28, R30).

`coming` pays because `T` ties two places and writes the thing once: `approaching` and two activations of the
thing are priced as `COMING` and one binding, a choice among every neuron, which the two places share.

**How `T` came to be.** Whatever came at the body stood in `approaching`'s residual at `(0, +2)` and, a frame
earlier, at `(0, +4)`: a same-to-same relation, one and the same neuron at both in each use, whatever it was
(D34, D47). The pick grew a function from it over `approaching` (D33), and the collapse tied the two places in
a parameter, `T`, since no one neuron there would pay as a fixed node (D27). Once `T` holds them they are
`T`'s, and no other function of the table names a thing at those coordinates.

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
2. **The variable holds, and the function with it.** `T`'s two places are two apart and a frame apart, and the
   same neuron stands at both; `approaching` stands two behind the first. `approaching` sends `coming` with
   `ball` in `T` (D31). The bid is accepted, covering `approaching` and binding both activations of the ball,
   and one level up `COMING` fires at `(10, 10)`, the body (§6.4).
3. **The call neuron speaks.** Its connection names the child of the function in `hit`'s table, one frame on
   at the body (D25). That is the entire record.
4. **Next frame the child is expanded and `hit` runs** (R28, R30). Its bid covers `COMING`, which connects to
   it once more, with what the hit earned (D10).

# 6. Wherever the two of them are

Put the body at `(50, 3)` and a ball at `(50, 5)` that was at `(50, 7)`. The subtraction gives the same two
offsets, the same variable holds, the same call neuron fires, and the same connection infers the child that hits. No coordinate ever reached the pattern (D11), so where they are costs nothing.

# 7. A thing never seen

A rock comes from ahead. It stands at both of `T`'s places, and for a parameter that is the whole test,
so it is `T`'s value on the spot and a member from then on (D27). The rock's own neuron has never been followed
by a hit and holds no such connection, so it has nothing to infer. `COMING` has: it fired for every ball and
fist that came straight on, and its lesson says `hit`. The rock is hit, the hit is rewarded, and the rock,
bound and open a frame later, connects to `hit` with that reward. Before `coming` existed, `approaching`'s own
habit under the cover would have answered the same way, since what a coverer with nothing to say covers speaks
in its place (R48).

So what generalizes is the situation written without the thing: the function's child, which says how something
came and leaves what it was to the variable, and it answers for every new thing.

# 8. Wherever it is relative to the body

That costs a function per direction. A ball coming from the left gives `(−2, 0)` now and `(−4, 0)` a frame
ago, an offset of `(−2, 0)` between them a frame apart, which is `T′`, and `coming-left` names `approaching`
two positions to its right; `COMING-LEFT` has learned the child that turns left. After the turn the ball is
ahead, `COMING` fires, and `hit` follows. The frame of reference
is never transformed: the body turns, the field is reported again, and the machine subtracts again. The radius
is what keeps the directions and distances to a handful: four positions each way, exact, and nothing beyond
(D1, D6).

A thing never seen that comes from the side is answered by `COMING-LEFT`, which has seen only turns pay. Where
that is wrong the reward says so, `COMING-LEFT`'s estimate for it turns negative, and the walk tries the next
action (R37).

# 9. The same thing as a reference frame

| in a reference-frame account | in the machine |
|---|---|
| the frame's origin | none in the pattern: a relation is an offset between two nodes (D34), and a child fires where its owner fired (D53) |
| a location in the frame | an offset from another node (D6, D34) |
| a feature at a location | a node in a relation (D16, D34) |
| the same object at any place in the world | one pattern, since no coordinate is part of a neuron (D11) and no pattern says where it stood (D13) |
| any object at that location | a class of a function at the offset (D41) |
| the same object at two locations | one parameter, the offset between them (D44) |
| what to do about it | the call neuron's connection to a function's child, which names no location at all |
