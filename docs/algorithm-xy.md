# An alternating pair

A worked case for [algorithm.md](algorithm.md): how `x, y, x, y, …` is seen by the two neurons in it, the
function it becomes, a cycle of two nodes entered at either letter, and what the stretch costs at every frame.
Then the same for other pairs, and the general function they generalize into. Nothing here is normative. It
follows the letters through grouping, the call at the present, the worth, and the generalization.

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
nodes, `x` and `y`, one relation, origin `y` (R50, D15). Its fit at frame 4 covers that one relation for one,
and is not taken (R24): `x` and `y` stand bare, and the function waits in the dictionary at a worth of −1
(D57).

**The cycle closes from the bare letters.** At frame 6 the pair fits again, at `y`, landing on the
`x` at 5. `(x, y, −1)` is at strength 2 and shares `x` with the fit, so it is a candidate; its other end, `y`,
is a node the function has, so it would close the cycle (R50, D15). The node it attaches at counts it.
`x`'s row holds `y` before it and `y` after it, so `x` counts its activations that hold `y` after together
with `y` before, the `x` at 3 and at 5: `with` 2. The `x` at 1 has `y` after it and nothing before, but the
relation closes the cycle, and where it does not land the lap ends, which is no failure, so its `without` is
nothing (D57, D54): worth 1. `y` after `x` is worth 2, `with` 3. The function's worth with the candidate is
the two worths less one for the run, 2, against −1 without it: a gain of 3, above the relation's contribution
alone, so it joins (D57, R54). The function is `y` after `x` and `x` after `y`, two nodes, two relations, a
cycle. Its latest node is in the cycle, so it has no fixed origin: a fit enters at either letter (D15, D54).

**The first call, over what the board holds.** Fit again at frame 6, it enters at the `y` and reads
back round the cycle while the closing relation lands (D54): `x` at 5, `y` at 4, and there the board ends,
since the `x` at 3 left it after frame 5 (D9). The lap ends without a failure.

| the fit at frame 6 | relations |
|---|---|
| A: the two relations among the three letters | 2 |
| B: one for the call | 1 |

It is taken. Its interneuron, `XY`, fires once, at the `y` at 6, at age 0, in the pose the fit read: from `y`
round to `y`, one lap (D53). It stands for the three letters, which leave the apex, each connecting
once more to it (D10). The `y` at 4 had seen the `x` at 3, but that activation is off the board, so the call
inherits nothing.

**Each letter from then on covers the call before it.** At frame 7 `x` fires and sees the call at `−1`, in its
pose (D5). The fit enters at `x`: `x` after `y` comes true on the call, since a fixed node comes true on a call
whose pose starts at that node (D54), and the lap stops there, since nothing lands on a relation another call
covers (D10).

| the fit at frame 7 | relations |
|---|---|
| A: the relation from `x` to the call, and the call's cost | 2 |
| B: one for the call | 1 |

It is taken. `XY` fires again, at the `x` at 7, in the pose from `x` round to `y`: the relation it read and
the old call's lap together, a lap and a half (D53, R23). The old call leaves the apex. At frame 8 the same
happens at `y`, two laps, and so on: the apex holds one call for the stretch, the latest, in a pose that
starts at `x` at `x` frames and at `y` at `y` frames and grows by half a lap each frame. The stretch costs one
at every frame, the standing call. The function costs 2 in the dictionary (D56, D13).

**After a break.** The stream falls silent and `x, y` comes again. At the first `y` the fit reads `y` after
`x` and reaches silence: one relation for one, not taken, and both letters stand bare for a frame. At the next
`x` the fit enters at `x`, reads `x` after `y` and `y` after `x` over the bare letters, two for one, and the
call fires, from `x` round to `x`, one lap. From the next `y` on each call covers the one before, as above.

**Other pairs.** `a, b, a, b, …` builds `AB` the same way, and `m, n, m, n, …` builds `MN`. Each is a function
of two nodes and two relations with its own interneuron, and none of them shares a node with another.

# 3. The call at the present

Each letter of the stretch the call fires anew at the present, at age 0, and infers from its connections what
follows (R36). Its connections are keyed by its pose, laps included (D25, D34), and a running cycle's call is
in a new pose every frame: from `y` round to `y` at one lap, from `x` to `y` at a lap and a half, from `y` to
`x` at two. Each is covered at age 1 by the call after it and connects once, to that call in its pose at `+1`
(D10): the call at a lap and a half holds the call at two laps and nothing else, since the letter under the
new call left the apex at age 0. So what a pose learns is the pose a half lap on, and that connection is as
strong as the stretches that reached it.

A call at a length some stretch reached before infers the next pose at `+1`. That call expands from the node
its pose starts at, the letter at the frame ahead, back over the letters that stand (R28), and the letter is
proposed (R49). A call at a length no stretch reached before has nothing for the frame ahead, and speaking
falls through the cover (R48): the letter it stands at speaks, and a base letter holds the other letter at
`+1`, or a call in a pose that expands to it, from the frames it stood bare. So the stretch is continued
either way, by the call where its length has recurred and by the letter under it where it has not. What the
call adds, at a length it knows, is one voter for the whole stretch where a letter knows only its
neighbor. One pose that holds what follows at every length is what a variable over laps will give, and it is
not in this case.

# 4. The worth

Nothing is kept for the function: its worth is read from the neurons at every level it fits at (R50,
D57). `y` counts `y` after `x`, `x` counts `x` after `y`, each over its own history. In a stretch `y` after
`x` holds at every `y`, so its `with` climbs and its `without` stays at zero. `x` after `y` holds at every `x`
but the first of each stretch, and there the lap ends, which is no failure, so its `without` stays at zero too.
Each relation is worth `with − 1`, and the function the two less one per run, since each call covers
the call before it and a run costs one in the end. A pair that stops coming sees its `with` slide out of the
two histories, the worth falls, and the function leaves (R18).

# 5. The general function

`XY`, `AB` and `MN` have the same shape: two nodes, two relations, offset 1 each way. Compared as graphs with
the names ignored, they pair from any node, since none has a fixed origin, and the pairing that lands every
relation is taken (R52): `y` with `b`, then along the relations `x` with `a`. At each node the two functions
hold different neurons, so each node is a parameter, and the general function is two open nodes, `P` and `P′`,
with `P′` after `P` and `P` after `P′` (D44). It costs 2 in the dictionary. Its call costs 3, one for the call
and two bindings, against 1 for a specific function's, so it replaces nothing: the specific functions stand
where they recur.

Where it pays is a pair never seen. `u, v, u, v, …` is read by it from the second letter, but a fit is taken
only while it covers more than it costs (R24), and the board holds three frames:

| the general function on `u, v, u, v, …` | covers | for |
|---|---|---|
| at the third letter | 2 | 3 |
| at the fourth, over the three on the board | 2 | 3 |

It is never taken over bare letters: the board never holds more than two of its relations at once. The
pair's own function, `{v, u@−1}`, is in the dictionary from the fourth letter, and at the sixth its cycle
closes as `XY`'s did and its call fires at 2 for 1. So in this stream the general function covers nothing,
and the dictionary holds it only for a stream where a pair's stretch is longer than the board and its own
function has not formed, which is where the variables case belongs.

# 6. The same thing as code

| in the code | in the machine |
|---|---|
| a loop that alternates two things | a function of two nodes with a cycle, one call on the apex for the loop |
| the loop body | the two relations |
| how many times it ran | the laps in the call's pose |
| the same loop over different things | one general function with two parameters, used where no specific one fits yet |
