# An alternating pair

A worked case for [algorithm.md](algorithm.md): how `x, y, x, y, …` is seen by the two neurons in it, the
function it becomes, a cycle of two nodes entered at either letter, and what the stretch costs at every frame.
Then the same for other pairs, and why nothing general is made of them. Nothing here is normative. It
follows the letters through grouping, the call at the present, the worth, and generalizing.

---

# 1. The stream

One channel, `letters`, laid out over time alone (D1): a letter sees the letter before it (D4), and its row
takes the letter after it a frame later (D18). The stream runs `x, y, x, y, …` for a stretch, then is silent
for longer than any reach, then runs `a, b, a, b, …`, then `m, n, m, n, …`, and so on. A pair's stretch may be
long or short, and a pair may come once or often. Nothing follows a stretch but silence, so what a pass teaches
is the alternation itself.

An activation stands on the board one frame beyond the farthest node any function reaches from its latest
(D9): while the dictionary holds pairs and cycles of offset 1, that is two frames, so at any frame the board
holds that frame and the two before it.

# 2. What the letters build

**The two relations.** `y` at frame 2 sees `x` one back: the relation `(y, x, −1)`, from the later neuron to
the earlier (D34). `x` at frame 3 sees `y` one back: `(x, y, −1)`. Each neuron counts its own (D47), and each
row holds the letter on both sides: `x` at 3 holds `y` at `−1` and, from frame 4, `y` at `+1`.

**The first function.** At frame 4, `(y, x, −1)` has strength 2 and no function holds it, so it starts one: two
nodes, `x` and `y`, one relation, origin `y` (R50, D15). Its fit at frame 4 covers that one relation for
nothing, and is taken (R24): its interneuron, `XY`, fires at the `y` at 4, at age 0, in the pose from `y` back
to `x` (D53), and the two letters leave the apex. The function is worth its occasions less one (D57).

**The cycle closes over the call.** At frame 5 `x` fires and sees the call one back, which the count reads as
the `y` under it (R50): the row of the `x` at 3 holds `y` one back, and the row of the `x` at 5 holds the call,
read as `y`, so `(x, y, −1)` is returned at strength 2 (D47), and it shares `x` with the function, so it is a
candidate; its other end, `y`, is a node the function has, so it would close the cycle (R50, D15). Both its
ends are nodes of the function, and the `x` at 5 returned it, so it attaches at `x`, not at the `y` read
through the call (R50). The node it attaches at counts it, over the rows that have taken what followed them,
which at frame 5 is the `x` at 3 alone: its row holds `y` before it and `y` after it, `with` 1 and `without`
0, worth 0. The `x` at 1 has `y` after it and nothing before, but the relation closes the cycle, and where it
does not land the lap ends, which is no failure, so its `without` is nothing (D57, D54). Alone, over the same
row, it is worth its strength less one, 0. A tie joins where it leaves fewer functions (R54); its strength as
returned is 2, so it does not leave for nothing (R18). The function is `y` after `x` and `x` after `y`, two
nodes, two relations, a cycle. Its latest node is in the cycle, so it has no fixed origin: a fit enters at
either letter (D15, D54).

**The call takes the lap.** A function that changed is fit again before the second election (R50). At frame 5
it enters at the `x` and reads `x` after `y` onto the `y` at 4, through the pair's call (D5). That call is a
call of this same function, and the relation is the function going round its cycle, so the call takes the lap
and no new call is made (D53).

| the fit at frame 5 | relations |
|---|---|
| A: the relation from `x` to the `y` under the call | 1 |
| B: nothing | 0 |

It is taken. `XY` stands at the `x` at 5, at age 0, the same call, in the pose from `x` round to `x`: one lap
(D53).

**Each letter from then on is a lap of the one call.** At frame 6 `y` fires and sees the call at `−1`, and
through it the `x` at 5 (D5). `y` after `x` is the call's own function going round, so the call takes it, and
`y` relates to nothing else there (R50).

| the fit at frame 6 | relations |
|---|---|
| A: the relation from `y` to the `x` under the call | 1 |
| B: nothing | 0 |

It is taken. `XY` stands at the `y` at 6, in the pose from `y` back to `x`, a lap and a half (D53). At frame 7
the same happens at `x`, two laps, and so on: the apex holds one call for the stretch, in a pose
that starts at `x` at `x` frames and at `y` at `y` frames and grows by half a lap each frame. The stretch
costs nothing at any frame: the standing call has no failure and no binding. The function costs 2 in the
dictionary (D56, D13).

**After a break.** The stream falls silent and `x, y` comes again. At the first `y` the fit reads `y` after
`x` and reaches silence: one relation for nothing, taken, and the call fires from `y` to `x`, half a lap. From
the next `x` on the call takes each letter as a lap, as above.

**Other pairs.** `a, b, a, b, …` builds `AB` the same way, and `m, n, m, n, …` builds `MN`. Each is a function
of two nodes and two relations with its own interneuron, and none of them shares a node with another.

# 3. The call at the present

Each letter of the stretch the call stands at the present, at age 0, and infers from its connections what
follows (R36). Its connections are keyed by its pose less its laps (D25): the node it started at and the node
it stands at. Standing at `x` it stood uncovered through the next frame and connected to what stood on the
apex there, itself standing at `y`, at `+1`; standing at `y`, to itself standing at `x`. Each is learned once
and strengthened at every letter after, whatever the length of the stretch.

So from its third letter on the call continues the stretch itself. The call it names at `+1` expands from the
node it stands at, the letter at the frame ahead, back over the letters that stand (R28), and the letter is
proposed (R49). Before that the call has nothing for the frame ahead, and speaking falls through the cover
(R48): the letter it stands at speaks, and a base letter holds the other letter at `+1` from the frames it
stood bare. What the call adds is one voter for the whole stretch where a letter knows only its neighbor.

# 4. The worth

Nothing is kept for the function: its worth is read from the neurons at every level it fits at (R50,
D57). `y` counts `y` after `x`, `x` counts `x` after `y`, each over its own history. In a stretch `y` after
`x` holds at every `y`, so its `with` climbs and its `without` stays at zero. `x` after `y` holds at every `x`
but the first of each stretch, and there the lap ends, which is no failure, so its `without` stays at zero too.
Each relation is worth `with − 1`, and the function the two together, since a call costs nothing of its own
(D13). A pair that stops coming sees its `with` slide out of the two histories, the worth falls, and the
function leaves (R18).

# 5. Generalizing

`XY`, `AB` and `MN` have the same shape: two nodes, two relations, offset 1 each way. Compared as graphs with
the names ignored, they pair from any node, since none has a fixed origin, and the pairing that lands every
relation is taken (R52): `y` with `b`, then along the relations `x` with `a`. At each node the two functions
hold different neurons, so each node would be a variable: two open nodes, `P` and `P′`, with `P′` after `P`
and `P` after `P′` (D44). Its call would cover two relations and pay two bindings, so it saves nothing at any
activation, where each pair's own function saves two. Over any history it saves less than the pairs'
functions save together, so it is not made, and each pair keeps its own function (R52, D57).

A pair never seen shows the same at the election. A call of two open nodes would pay two bindings, and a fit
is taken only while it covers more than it costs (R24). On `u, v, u, v, …` the board holds three frames:

| two open nodes on `u, v, u, v, …` | covers | for |
|---|---|---|
| at the third letter | 2 | 2 |
| at the fourth, over the three on the board | 2 | 2 |

It would never be taken over bare letters: the board never holds more than two of its relations at once, and
two bindings cost as much. The pair's own function, `{v, u@−1}`, is in the dictionary from the fourth letter
and taken there, 1 for nothing, and at the fifth its cycle closes as `XY`'s did. So a new pair is learned as
`x, y` was, from its own letters.

# 6. The same thing as code

| in the code | in the machine |
|---|---|
| a loop that alternates two things | a function of two nodes with a cycle, one call on the apex for the loop |
| the loop body | the two relations |
| how many times it ran | the laps in the call's pose |
| the same loop over different things | a function per pair: one function over two open nodes would pay two bindings for two relations and save nothing (R52) |
