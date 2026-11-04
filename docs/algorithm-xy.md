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

# 2. What the letters build

**The two relations.** `y` at frame 2 sees `x` one back: the relation `(y, x, −1)`, from the later neuron to
the earlier (D34). `x` at frame 3 sees `y` one back: `(x, y, −1)`. Each neuron counts its own (D47), and each
row holds the letter on both sides: `x` at 3 holds `y` at `−1` and, from frame 4, `y` at `+1`.

**The first function.** At frame 4, `(y, x, −1)` has strength 2 and no function holds it, so it starts one: two
nodes, `x` and `y`, one relation, origin `y` (R50, D15). Its fit at frame 4 covers that one relation for the
call's own relation, one for one, and is not taken (R24): `x` and `y` stand bare, and the function waits in
the dictionary at a worth of −1 (D57).

**The cycle closes from the bare letters.** At frame 6 the pair's fit is placed again, at `y`, landing on the
`x` at 5. `(x, y, −1)` is at strength 2 and shares `x` with the fit, so it is a candidate; its other end, `y`,
is a node the function has, so it would close the cycle (R50, D15). The node it attaches at answers for it.
`x`'s row holds `y` before it and `y` after it, so `x` counts its activations that hold `y` after together
with `y` before, the `x` at 3 and at 5: `with` 2. The `x` at 1 has `y` after it and nothing before, but the
relation closes the cycle, and where it does not land the lap ends, which is no failure, so its `without` is
nothing (D57, D54): worth 1. `y` after `x` is worth 2, `with` 3. The function's worth with the candidate is the
two worths less one for the run, 2, against −1 without it: a gain of 3, above the relation's contribution
alone, so it joins (D57, R54). The function is `y` after `x` and `x` after `y`, two nodes, two relations, a cycle. Its
latest node is in the cycle, so it has no fixed origin: a fit enters at either letter (D15, D54).

**One call for the stretch, from the sixth letter.** Placed again at frame 6, the fit enters at the `y` and
reads back round the cycle while the closing relation lands (D54): `x` at 5, `y` at 4, `x` at 3, `y` at 2, `x`
at 1, and silence before it, where the lap ends without a failure.

| the fit at frame 6 | relations |
|---|---|
| A: the five relations among the six letters | 5 |
| B: the call's own relation | 1 |

It is taken. Its interneuron, `XY`, fires at the fit's two ends (D53): its start at the `y` at 6, at age 0,
and its end at the `x` at 1, at age 5, with the relation between them, `−5`, its own, five laps of one (D37).
The two letters at the ends are replaced by the call's activations; the four between are gone with their
relations (D10). All six leave the apex.

**Each letter from then on covers the call before it.** At frame 7 `x` fires and sees the call at `−1`, standing
at the `y`'s place (D5). The fit enters at `x`: `x` after `y` comes true on the call's start, since a fixed
node comes true on an end of a call that replaced an activation of that neuron (D54), and the lap stops there,
since nothing lands on an activation between another call's ends (D10).

| the fit at frame 7 | relations |
|---|---|
| A: the relation from `x` to the call, and the call's count | 2 |
| B: the call's own relation | 1 |

It is taken. `XY` fires again, its start at the `x` at 7, and takes the earlier call's end as its own: the end
of the stretch stays the one activation at frame 1, and the new call's own relation is `−6` (D53, R23). The
earlier call's two activations leave the apex. At frame 8 the same happens at `y`, and so on: the apex holds
one call for the stretch, the latest, and it fires at `x` frames and `y` frames alike. The stretch costs one
relation at every frame, the standing call's own. The function costs 2 in the dictionary (D56, D13).

**After a break.** The stream falls silent and `x, y` comes again. At the first `y` the fit reads `y` after
`x` and reaches silence: one relation for one, not taken, and both letters stand bare for a frame. At the next
`x` the fit enters at `x`, reads `x` after `y` and `y` after `x` over the bare letters, two for one, and the
call fires, its end at that first `x`. From the next `y` on each call covers the one before, as above.

**Other pairs.** `a, b, a, b, …` builds `AB` the same way, and `m, n, m, n, …` builds `MN`. Each is a function
of two nodes and two relations with its own interneuron, and none of them shares a node with another.

# 3. The call at the present

Each letter of the stretch the call fires anew at the present, at age 0, and infers from its connections what
follows (R36): the next letter, which its earlier activations saw on the apex at offset 1, and after the
stretch, the silence. The call it covered is gone from the apex, and the new one votes, from its start, since
an end never stands at the present and infers nothing (R46, D9). The letter that just fired is replaced by the
call's start, writes one connection to the call, and is silent (D10). So the call speaks for the stretch: at
the base, a letter's own connections say only that `x` is followed by `y` and `y` by `x`, which the call says
already.

# 4. The worth

Nothing is kept for the function: its worth is read from the neurons at every level it is placed at (R50,
D57). `y` answers for `y` after `x`, `x` for `x` after `y`, each over its own history. In a stretch `y` after
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
with `P′` after `P` and `P` after `P′` (D44). It costs 2 in the dictionary. Its call costs 3, its own relation
and two bindings, against 1 for a specific function's, so it replaces nothing: the specific functions stand
where they recur.

Where it pays is a pair never seen. `u, v, u, v, …` is read by it from the second letter, but a fit is taken
only while it covers more than it costs (R24):

| the general function on `u, v, u, v, …` | covers | for |
|---|---|---|
| at the third letter | 2 | 3 |
| at the fourth | 3 | 3 |
| at the fifth, `u` | 4 | 3 |

At the fifth letter it is taken: its call stands with `u` bound in `P` and `v` in `P′`, its start at that `u`
and its end at the first. The pair's own function, `{v, u@−1}`, has been in the dictionary since the fourth
letter, never placed, and at the sixth letter its cycle closes as `XY`'s did. Placed again there, it enters at
`v`, lands on the general call's start as the `u` before it, and covers the call's whole count and the
relation to it, 4 for 1 (D22). It is taken, the general call's two activations leave the apex, and from then on
the pair's own cycle covers the stretch at 2 for 1 a frame. So the general function covers one frame of each
pair that comes for the first time, and nothing else in this stream. It would cover more where a pair comes
round too rarely for its own function to form within a history, or where the stream showed a pair twice and
never again.

# 6. The same thing as code

| in the code | in the machine |
|---|---|
| a loop that alternates two things | a function of two nodes with a cycle, one call on the apex for the loop |
| the loop body | the two relations |
| how many times it ran | the relation between the call's two ends, over the lap's offset of two |
| the same loop over different things | one general function with two parameters, used where no specific one fits yet |
