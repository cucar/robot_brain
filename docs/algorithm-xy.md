# An alternating pair

A worked case for [algorithm.md](algorithm.md): how `x, y, x, y, …` is seen by the two neurons in it, the two
functions it becomes, each a cycle of two nodes with its own origin, and what one call over the whole stretch
costs. Then the same for other pairs, and the general function they generalize into. Nothing here is normative.
It follows the letters through grouping, the call at the present, the count, and the generalization.

---

# 1. The stream

One channel, `letters`, laid out over time alone (D1): a letter sees the letter before it (D4). The stream runs
`x, y, x, y, …` for a stretch, then is silent for longer than any reach, then runs `a, b, a, b, …`, then
`m, n, m, n, …`, and so on. A pair's stretch may be long or short, and a pair may come once or often. Nothing
follows a stretch but silence, so what a pass teaches is the alternation itself.

# 2. What the letters build

**The two relations.** `y` at frame 2 sees `x` one back: the relation `(y, x, −1)`, from the later neuron to
the earlier (D34). `x` at frame 3 sees `y` one back: `(x, y, −1)`. Each neuron counts its own (D47), and from
frame 4 on each relation has come round twice (R21).

**The first function.** At frame 4, `(y, x, −1)` has strength 2 and no function holds it, so it starts one: two
nodes, `x` and `y`, one relation, origin `y` (R50, D15). It is placed at frame 4, where it lands on `y` at 4
and `x` at 3. It covers the relation and both names, 3, for a call of 1, and is accepted (R24). Its
interneuron, `XY`, fires at frame 4, the origin, covering frames 3 and 4 (D53).

**The cycle closes.** At frame 5, `x` fires, and its neuron returns `(x, y, −1)`, strength 2. That relation
shares `y` with the origin of the call `XY` fired a frame before, which is still open, so it is a candidate of
`XY`'s function, attached at that placement (R50). `XY` is placed at every pair `x, y` in the stretch, and `x`
follows every one of them, so the candidate is attached at every accepted placement, and it joins: the
function is now `y` to `x` at −1 and `x` to `y` at −1, two nodes, two relations, a cycle (D15, R54). The
relation a candidate would add from `x` to `x` at −2 is implied by the other two along the cycle, and is never
held (D34).

**Two functions, one per origin.** `XY` has origin `y`, so it is recognized at the `y` frames, and a fit of it
ends on `y`. At an `x` frame the same two relations with origin `x` are another function, `YX`, started the
same way when `(x, y, −1)` first recurred and grown into the cycle by the same join (D15). Each has its own
interneuron.

**One call over the stretch, at any moment.** From then on the stretch is recognized at every letter. At a `y`
frame `XY` fits, reading back round the cycle as far as the relations keep coming true, to the start of the
stretch (D54); at an `x` frame `YX` fits the same way. The call fires at the present letter, covers everything
back to the start, and takes back the call that fired at the letter before, which covered one letter less
(R23). So the letter that just fired is covered at age 0, the origin, and never stands on the apex; the apex
holds one call for the stretch, the latest (D49, D10). The call counts how far back it ran (D12).

| a stretch of `2k` letters, at its last letter | numbers |
|---|---|
| bare: `2k` names and `2k − 1` relations | `4k − 1` |
| the call: its name, and how far back round the cycle it ran | 2 |

Each function costs 4 in the dictionary, two names and two offsets (D56, D13); the pair of them, 8.

**Other pairs.** `a, b, a, b, …` builds `AB` and `BA` the same way, and `m, n, m, n, …` builds `MN` and `NM`.
Each is a function of two nodes and two relations with its own interneuron, and none of them shares a node
with another.

# 3. The call at the present

Each frame of the stretch the call fires anew at the present letter, at age 0, and infers from its connections
what follows (R36): the next letter, which its earlier activations saw on the apex at offset 1 as the call
that fired there, and after the stretch, the silence. The activation it took back is gone from the apex, and
the new one votes (R46). The letter that just fired is covered at age 0 and writes one connection, to the
call, and is silent (D10). So the call speaks for the stretch: at the base, a letter's own connections say
only that `x` is followed by `y` and `y` by `x`, which the call says already, and nothing is inferred at the
base that the call does not say.

# 4. The count

A function's counts are over its accepted placements (R50, D57). Each letter of a stretch is one placement of
the function with that origin, both relations come true at it, and nothing fails: a fit reads back only as far
as the relations hold. Over a window of placements each relation is worth `2 · held − 2`, with `held` the
placements, so a pair that comes round at all keeps its two functions, and one that stops coming sees their
placements age out and both pruned (R18).

# 5. The general function

`XY`, `AB` and `MN` have the same shape: two nodes, two relations, offset 1 each way. Compared as graphs with
the names ignored, they pair from their origins (R52): `y` with `b`, then along the relations `x` with `a`. At
each node the two functions hold different neurons, so each node is a parameter, and the general function is
two open nodes, `P` and `P′`, with `P′` after `P` and `P` after `P′`, origin `P` (D44). `YX`, `BA` and `NM`
generalize the same way into its twin, origin `P′`. Each costs 4 in the dictionary and replaces nothing: the
specific functions stand where they recur, since a call of one of them costs 2 against the general function's
4, its name, how far it ran, and two values.

Where the general function pays is a pair never seen. `u, v, u, v, …` fits it at the second `v`, `u` bound in
`P′` and `v` in `P`, and from then on every letter of the stretch is covered by one of the two general
functions at 4 numbers, against `4k − 1` bare, with nothing learned about `u` or `v` first. Each becomes a
member of its parameter in that pass. If the pair keeps coming, `(v, u, −1)` recurs, its own function starts,
grows into the cycle, and covers the pair for 2 from then on; the general function goes back to waiting for
the next new pair.

# 6. The same thing as code

| in the code | in the machine |
|---|---|
| a loop that alternates two things | a function of two nodes with a cycle, one call for the whole loop |
| the loop body | the two relations |
| how many times it ran | one number the call counts |
| the same loop over different things | one general function with two parameters, used where no specific one fits |
