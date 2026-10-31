# Hit whatever comes at you, as a machine that could be built

A worked case for [algorithm.md](algorithm.md): a machine that hits whatever is coming at it, wherever the two
of them are, written out neuron by neuron. Nothing here is normative, and the machine is shown running: its
dictionary and its connections are as listed, and its histories already hold moments like this one. It shows
where the design does its reference-frame calculation: the machine subtracts coordinates when it assembles a
neighborhood (D6), and a function is written in the offsets between its own nodes (D34), so everything that is
learned or acted on is written in differences, and a thing's course is written without saying where it was.
And it shows what answers for a thing never seen before: the situation written without the thing.

---

# 1. The environment

**In.** A field, laid out over two activation dimensions and time. What is in it is reported as events: `ball`
where a ball is, `fist` where a fist is. One more event is the world's own: `approaching` fires at the body's
position in every frame in which something is coming at the body, whatever it is.

**Out.** One action dimension with three base actions, none with an argument: `hit`, which strikes what is
straight ahead of the body, `turn-left` and `turn-right`.

# 2. What stands under them

The field declares a radius of `4` in each of its two spatial dimensions and `1` in time (D1), so `approaching`,
`ball` and `fist`, base events all, see four positions in every direction and one frame back (D4), and within
that reach they are each other's neighbors on the apex (D5). Within the radius every distance is exact (D6);
the distances this case uses are `2` and `4`.

The layout has two spatial dimensions, so a bare record is four numbers: the neuron, the frame and two
positions (D13). A call's record is four, then one per relation that did not come true and one per value
(D12, D22). Offsets are written `(time, x, y)`.

# 3. The neurons

| Kind | Neurons |
|---|---|
| events | `approaching`, `ball`, `fist`, and whatever else the field holds |
| the functions | `coming`, `coming-left` and `coming-right`, each naming `approaching` and a parameter, `T`, `T′` or `T″`, at two nodes a frame apart that hold one and the same thing, one function per course (D44); their interneurons `COMING`, `COMING-LEFT` and `COMING-RIGHT` (D48) |
| base actions | `hit`, `turn-left`, `turn-right` |
| the functions of the actions | one per action: the action and the interneuron that stood a frame before it, `{hit, COMING@(−1, 0, −2)}` and its interneuron `COMING·hit`, and likewise for the turns |

# 4. The parameters, and what learns

Each parameter ties two nodes, the same thing at both, whatever it is (D38): one thing, two positions nearer
than it was a frame ago. A parameter is written in the relations of its nodes (D44), so it says how the thing
moved and not where it was. The function it belongs to names `approaching`, which fires at the body, and that
says where.

| Parameter | where its function puts it, from `approaching` | its other node |
|---|---|---|
| `T`, coming straight on | `(0, 0, +2)` | `(−1, 0, +4)` |
| `T′`, from the left | `(0, −2, 0)` | `(−1, −4, 0)` |
| `T″`, from the right | `(0, +2, 0)` | `(−1, +4, 0)` |

`coming` is five relations: `T` at time `0`, at time `−1`, at `y +2`, at `y +4`, and at `x 0`. A relation
lands on one activation (D15), and the one at `x 0` lands on the ball a frame ago, the earlier coordinate
(D31). The ball now has its `x` fixed by no relation, so the call writes it. Where a ball stands two ahead of
the body and stood four ahead a frame ago, the relations land, `T` is `ball`, and the election takes the
function: it covers `approaching` and both activations of the ball, twelve numbers, for a record of six: four,
the value, and the ball's `x`. `COMING` fires at the
centroid of what it covers and binds, `(0, +2)` from the body, in the frame `approaching` fires (D53), and
the ball stays where it was, bound (D10). `COMING` is anything coming straight on, and it holds the lesson: it
connects to what follows it with what the hit earned (R31). The thing itself, a `ball` or a `fist`, connects
too, but as the base neuron it is, pooled over every place a ball has ever stood (D11); what it learns is a
ball's habit, not a ball coming straight on.

**The function of the action.** `hit` runs at the body a frame later, and sees `COMING` at `(−1, 0, −2)`.
Where that recurs, grouping gives `{hit, COMING@(−1, 0, −2)}`: three relations, an entry of five. It covers
`hit` and `COMING`'s record, ten numbers, for four, and its interneuron `COMING·hit` fires at `(0, +1)` from
the body. In that frame `COMING` is covered, and connects once to `COMING·hit` at `(+1, 0, −1)`, with that
frame's reward (D10). Inferred from `COMING` at its age 0, `COMING·hit` is placed a frame ahead at `(0, +1)`
and expanded: `hit` at the body and `COMING` where it stands (R28). The match credits `COMING`'s record and
charges nothing (R49), so `hit` is proposed, and output (R30).

**A lesson per thing on a course** is one level up. The bound ball stands at `COMING`'s own coordinate, a
neighbor at offset zero (D26). Its positions are covered, and its neuron is written as `COMING`'s value, so a
function `{COMING, ball@(0, 0, 0)}` covers that one number and `COMING`'s record, seven, for four: three
relations and two neurons, an entry of five, so its margin is positive from its second row (D57). Until then the first
ball is answered by `COMING`, not by a lesson of its own.

**How `T` came to be.** A ball stood in `approaching`'s rows at `(0, 0, +2)` and at `(−1, 0, +4)`, and
grouping merged those relations where they recurred (R50): `coming-ball`, `{approaching, ball@(0, 0, +2),
ball@(−1, 0, +4)}`, five relations and two neurons, an entry of seven, saving seven per row. A fist did the
same, `coming-fist`. When the second of the two was grouped, find read its relations with the thing's end open
and reached the first (R53). The two differ only at nodes that hold one neuron within each function, so those
nodes are one parameter (R52): `coming`, with `T`, an entry of six, a record of six. It is priced as a
replacement of the two: two entries of seven and five per row, fourteen and `5n`, against one entry of six
and six per row, six and `6n`, `n` being the rows of `approaching`'s history that hold a ball or a fist coming
straight on. `coming` joins while `14 + 5n > 6 + 6n`, fewer than eight such rows, and with each further thing
that has come straight on twice the specific side gains seven more. So `coming` stands where things of
several kinds come, or the history is short; where one or two things come often, their own functions are
cheaper, and stand instead. The case shows the machine with `coming` in the dictionary.

**The habit.** Before `coming` existed, `approaching` stood uncovered while things came and `hit` ran, so it
holds a connection of its own: after `approaching`, `hit`, with what those hits earned. Once `coming` holds,
`approaching` is covered in the frame it fires and connects once more, to `COMING` (D10). What it wrote
stands, since nothing leaves a connection (R31).

# 5. One moment, worked

The body is at `(10, 10)`. A ball is at `(10, 12)`, and a frame ago it was at `(10, 14)`. `approaching` fires
at `(10, 10)`.

1. **The machine subtracts.** Assembling the neighborhood of the `approaching` activation, it takes the
   difference of coordinates in each dimension (D6): the ball now is at `(0, 0, +2)`, the ball a frame ago at
   `(−1, 0, +4)`.
2. **The relations land, and the function with them.** `approaching`'s activate call returns its relations to
   the ball, each held in one row of its history or more (R21), and find reads them with the ball's end open
   and reaches `coming` (R53). Placed, every relation of `coming` comes true with `T` as `ball` (D31). The
   election takes it, twelve numbers for six (R24), and at the next level `COMING` fires at `(10, 12)`, the
   centroid of `(10, 10)`, `(10, 12)` and `(10, 14)` (§6.4).
3. **The call neuron speaks.** `COMING`'s connection names `COMING·hit` one frame on at `(+1, 0, −1)` (D25).
   Expanded, that is `hit` at the body. The match holds, and `hit` is proposed and output (R49, R47).
4. **Next frame `hit` runs.** `{hit, COMING@(−1, 0, −2)}` covers `hit` and `COMING`, and `COMING` connects
   once more to `COMING·hit`, with what the hit earned (D10).

# 6. Wherever the two of them are

Put the body at `(50, 3)` and a ball at `(50, 5)` that was at `(50, 7)`. The subtraction gives the same two
offsets, the same relations land, the same interneuron fires at `(50, 5)`, and the same connection infers the
call that hits. No coordinate ever reached the function (D11), so where they are costs nothing.

# 7. A thing never seen

A rock comes from ahead for the first time. `approaching`'s row holds its relations to the rock, at
`(0, 0, +2)` and `(−1, 0, +4)`, each in one row only, so nothing is grouped from them (R50). They are still
returned, and find reads them with the rock's end open (R53): `(approaching, ?, y +2)` and the rest reach
`coming`, since a parameter stands at that end. `coming` fits with `T` as `rock`: a parameter takes any neuron,
and one that was not a member is one from then on (D44, D54). So `COMING` fires. The rock's own neuron has
never been followed by a hit and holds no such connection. `COMING` has: it fired for every ball and fist that
came straight on, and its lesson says `hit`. The rock is hit, the hit is rewarded, and the rock, bound and open
a frame later, connects to `COMING·hit` with that reward.

Where `coming` is not in the dictionary, the rock's relations reach `coming-ball` and `coming-fist` as partial
matches whose every relation fails, nothing covers the rock, and `approaching` stands bare and answers from
its own habit (R35). The rock gets a function of its own at its second approach, and `coming` is priced again
as a replacement of the three.

So what generalizes is the situation written without the thing: the function's interneuron, which says how
something came and leaves what it was to the parameter, and it answers for every new thing.

# 8. Wherever it is relative to the body

That costs a function per direction. A ball coming from the left gives `(0, −2, 0)` now and `(−1, −4, 0)` a
frame ago. Those are different offsets, so they are different relations, and `coming-left` is a function of
its own, `{approaching, T′@(0, −2, 0), T′@(−1, −4, 0)}`, reached by nothing `coming` writes. `COMING-LEFT`
fires at `(−2, 0)` from the body and has learned the call that turns left. After the turn the ball is ahead,
`COMING` fires, and `hit` follows. The frame of reference is never transformed: the body turns, the field is
reported again, and the machine subtracts again. The radius is what keeps the directions and distances to a
handful: four positions each way, exact, and nothing beyond (D1, D6).

A thing never seen that comes from the side is answered by `COMING-LEFT`, which has seen only turns pay. Where
that is wrong the reward says so, `COMING-LEFT`'s estimate for it turns negative, and the walk wires the next
action (R37).

# 9. The same thing as a reference frame

| in a reference-frame account | in the machine |
|---|---|
| the frame's origin | none in the function: a relation is an offset between two nodes (D34), and an interneuron fires at the centroid of what it covers (D53) |
| a location in the frame | an offset from another node (D6, D34) |
| a feature at a location | a node in a relation (D16, D34) |
| the same object at any place in the world | one function, since no coordinate is part of a neuron (D11) and no function says where it stood (D13) |
| any object at that location | a parameter's node at the offset, open on what stands there (D44) |
| the same object at two locations | one parameter, the relations between its nodes (D44) |
| what to do about it | the interneuron's connection to the action's call, which names no location at all |
