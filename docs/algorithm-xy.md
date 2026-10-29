# An alternating pair

A worked case for [algorithm.md](algorithm.md): how `x, y, x, y, …` is seen by the two neurons in it, the
function it becomes, a cycle of two nodes, and what one call over the whole run costs. Then the same for other
pairs, and the one thing the case does not settle: how the cycles of different pairs are generalized into one
function. Nothing here is normative. It follows the letters through grouping, the call at the present, the count, and
the question left open.

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

**One call over the stretch, at any moment.** From then on the function is recognized at every letter of the
stretch. At each frame the fit's origin is the letter that just fired, and the fit reads back round the cycle
as far as the relations keep coming true, to the start of the stretch (D54). The call fires at the present
letter, covers everything back to the start, and takes back the call that fired at the letter before, which
covered one letter less (R23). So the letter that just fired is covered at age 0, the origin, and never stands
on the apex; the apex holds one call for the stretch, the latest (D49, D10). The call counts how far back it
ran (D12).

| a stretch of `2k` letters, at its last letter | numbers |
|---|---|
| bare: `2k` names and `2k − 1` relations | `4k − 1` |
| the call: its name, and how far back round the cycle it ran | 2 |

The function costs 4 in the dictionary, two names and two offsets (D56, D13).

**Other pairs.** `a, b, a, b, …` builds `AB` the same way, and `m, n, m, n, …` builds `MN`. Each is a function of
two nodes and two relations, each has its own interneuron, and none of them shares a node with another.

# 3. The call at the present

Each frame of the stretch the call fires anew at the present letter, at age 0, and infers from its connections
what follows (R36): the next letter, which its earlier activations saw on the apex at offset 1 as the call
that fired there, and after the stretch, the silence. The activation it took back is gone from the apex, and
the new one votes (R46). The letter that just fired is covered at age 0 and writes one connection, to the
call, and is silent (D10). So the call speaks for the stretch: at the base, a letter's own connections say
only that `x` is followed by `y` and `y` by `x`, which the call says already, and nothing is inferred at the
base that the call does not say.

# 4. The count

The function's counts are over its accepted placements (R50, D57). Each letter of a stretch is one placement,
the two relations come true once per traversal back from it, and nothing fails: a fit reads back only as far
as the relations hold. Over a window of placements each relation is worth `2 · held − 2`, with `held` the
traversals, so a pair that comes round at all keeps its function, and one that stops coming sees its placements
age out and its function pruned (R18).

# 5. What is not settled

`XY`, `AB` and `MN` have the same shape: two nodes, two relations, offset 1 each way. Compared as graphs with
the names ignored, their nodes pair (R52), and at each node the two functions hold different neurons. If the
pairing were fixed, one general function would follow: two open nodes and the two relations, whatever the pair,
and a stretch of a pair never seen would be covered on its first pass, its two letters bound as values, for 2
numbers and two values against `4k − 1` bare.

What is not settled is the pairing. A cycle of two nodes can be read from either node, so `x` pairs with `a` or
with `b`, and the two readings give two different general functions. Nothing in the rule for generalizing says
which, and a cycle read from the wrong node generalizes `x` with `b` and `y` with `a`, which fits the same
streams and says nothing different. Whether that is one function or two, and what the variable at a node that
is reached again stands for across functions, is left open here, and the design marks it so (R52).

# 6. The same thing as code

| in the code | in the machine |
|---|---|
| a loop that alternates two things | a function of two nodes with a cycle, one call for the whole loop |
| the loop body | the two relations |
| how many times it ran | one number the call counts |
| the same loop over different things | functions of the same shape, not yet generalized |
