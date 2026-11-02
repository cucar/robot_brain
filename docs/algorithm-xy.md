# An alternating pair

A worked case for [algorithm.md](algorithm.md): how `x, y, x, y, …` is seen by the two neurons in it, the
function it becomes, a cycle of two nodes with one origin, and what the stretch costs at every frame. Then
the same for other pairs, and the general function they generalize into. Nothing here is normative. It
follows the letters through grouping, the call at the present, the worth, and the generalization.

---

# 1. The stream

One channel, `letters`, laid out over time alone (D1): a letter sees the letter before it (D4). The stream runs
`x, y, x, y, …` for a stretch, then is silent for longer than any reach, then runs `a, b, a, b, …`, then
`m, n, m, n, …`, and so on. A pair's stretch may be long or short, and a pair may come once or often. Nothing
follows a stretch but silence, so what a pass teaches is the alternation itself.

# 2. What the letters build

**The two relations.** `y` at frame 2 sees `x` one back: the relation `(y, x, −1)`, from the later neuron to
the earlier (D34). `x` at frame 3 sees `y` one back: `(x, y, −1)`. Each neuron counts its own (D47).

**The first function.** At frame 4, `(y, x, −1)` has strength 2 and no function holds it, so it starts one: two
nodes, `x` and `y`, one relation, origin `y` (R50, D15). It is placed at frame 4, where it lands on `y` at 4
and `x` at 3. It covers the relation, one relation, for a call that costs nothing, and is accepted (R24). Its
interneuron, `XY`, fires at frame 4, the origin, covering frames 3 and 4 (D53).

**The pair covers the letter after it from view.** At frame 5, `x` fires and sees the call `XY` at `−1` and
nothing under it: the call stands at the `y`'s place and is the neighbor there, and the `x` at 3 is beyond
its reach (D5). Its neuron returns `(x, XY, −1)` at
strength 1, and `x` stands bare for the frame. At frame 6, `y` sees that bare `x`, `XY` fits again, and fires
at 6 covering frames 5 and 6.

**The cycle closes through the call.** At frame 7, `x` sees `XY` at `−1` again, and `(x, XY, −1)` has strength
2. The call's function has `y` as its origin, and `y` is a node of that same function, so the relation is read
as `x` after `y` (R50, D54): a candidate attached at `y` whose other end, `x`, is already a node. It closes the
cycle (D15). `x` answers for it over its history, counting a row as holding `y` one back where it holds `y`
or a call whose origin is `y` (R50): the `x` at 3 saw `y` bare, the ones at 5 and 7 saw the call, so `with`
is 3 and `without` 0. It costs one relation in the dictionary and would save the relation between its own call and
`XY` at each of the three, so its worth is 5 and it joins (D57, R54). The function is now `y` after `x` and
`x` after `y`, two nodes, two relations, a cycle. The relation is in that function and starts none of its own
(R50), so nothing forms over the call at the `x` frames. Placed again at frame 7 the function has no fit,
since no `y` stands at the present (D31):
`x` is bare for one frame.

**One call per lap, each covering the one before.** At frame 8 the cycle fits at `y`: `y` after `x` lands on
the `x` at 7, and `x` after `y` reaches the `y` at 6, which stands as the call `XY` fired there. A fixed node
comes true on a call whose origin is that neuron, so the fit lands on the call and covers its whole count
(D54, D22). The `y` at 6 is covered, so the fit reaches it only as the call standing there: at the base
reach the fit reads one lap and lands on the call of the lap before. The call counts nothing for the lap; how
many it covers is in what the calls cover (D24).

| the fit at frame 8 | relations |
|---|---|
| A: the two relations, `y` to `x` and `x` to the call at 6, and the call's count, nothing | 2 |
| B: the call's count, no failure and no argument | 0 |

It is accepted, `XY` fires at 8, and the call at 6 is covered and leaves the apex (D10). From then on the
stretch is recognized at every `y`: the call fires there, covers the lap and the call before it, and the apex
holds one call for the stretch, the latest (D49). The `x` between stands on the apex for a frame, with its
relation back to the call, one relation, and is covered at the next `y`. The laps are in the calls, each under the
one after it, and expansion follows them back (R28). The stretch costs nothing at a `y` and one relation at an
`x`. The function costs 2 in the dictionary, its two relations (D56, D13).

**After a break.** The stream falls silent and `x, y` comes again. At the first `y` the fit reads `y` after
`x`, and `x` after `y` reaches silence. The relation that closes a cycle is never a failure: the fit ends
there, a lap read once is the cycle run once (D54), so it covers one relation for nothing and is taken, and the
call fires. At the next `y` the closing relation lands on that call through its origin, as in a run that
never stopped. Nothing had to stay on the board for it but the call.

**Other pairs.** `a, b, a, b, …` builds `AB` the same way, and `m, n, m, n, …` builds `MN`. Each is a function
of two nodes and two relations with its own interneuron, and none of them shares a node with another.

# 3. The call at the present

Each `y` of the stretch the call fires anew at the present letter, at age 0, and infers from its connections
what follows (R36): the next letter, which its earlier activations saw on the apex at offset 1, and after the
stretch, the silence. The call it covered is gone from the apex, and the new one votes (R46). The letter that
just fired is covered at age 0, writes one connection to the call, and is silent (D10). So the call speaks for
the stretch: at the base, a letter's own connections say only that `x` is followed by `y` and `y` by `x`,
which the call says already.

# 4. The worth

Nothing is kept for the function: its worth is read from the neurons at every level it is placed at (R50,
D57). `y` answers for `y` after `x`, `x` for `x` after `y`, each over its own history. In a stretch both hold
at every lap, so `with` climbs and `without` stays at zero: each relation is worth `2 · with − 1`. A pair that
stops coming sees its `with` slide out of the two histories, the worth falls below zero, and the function
leaves (R18).

# 5. The general function

`XY`, `AB` and `MN` have the same shape: two nodes, two relations, offset 1 each way. Compared as graphs with
the names ignored, they pair from their origins (R52): `y` with `b`, then along the relations `x` with `a`. At
each node the two functions hold different neurons, so each node is a parameter, and the general function is
two open nodes, `P` and `P′`, with `P′` after `P` and `P` after `P′`, origin `P` (D44). It costs 2 in the
dictionary and replaces nothing: the specific functions stand where they recur, since a call of one of them
costs nothing against the general function's two bindings.

Where it would pay is a pair never seen. `u, v, u, v, …` fits it at the second `v`, `u` bound in `P′` and `v`
in `P`, reading back over the four bare letters: 3 for 2. But `(v, u, −1)` has strength 2 at that same frame,
the pair's own function starts and is placed beside it, 1 for 0, and a fit priced at nothing is taken first
(R24, D28). So in this stream the general function covers nothing. It would, where a pair comes round too rarely
for its own function to form within a history, or where the stream showed a pair twice and never again.

# 6. The same thing as code

| in the code | in the machine |
|---|---|
| a loop that alternates two things | a function of two nodes with a cycle, one call on the apex for the loop |
| the loop body | the two relations |
| how many times it ran | not written anywhere: one lap per call at the base reach, the earlier laps in the calls under it |
| the same loop over different things | one general function with two parameters, used where no specific one fits |
