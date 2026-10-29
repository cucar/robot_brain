# An alternating pair

A worked case for [algorithm.md](algorithm.md): how `x, y, x, y, …` is seen by the two neurons in it, the
function it becomes, a cycle of two nodes, and what one call over the whole run costs. Then the same for other
pairs, and the one thing the case does not settle: how the cycles of different pairs are generalized into one
function. Nothing here is normative. It follows the letters through grouping, the standing call, the count, and
the question left open.

---

# 1. The stream

One channel, `letters`, laid out over time alone (D1): a letter sees the letter before it (D4). The stream runs
`x, y, x, y, …` for a stretch, then is silent for longer than any reach, then runs `a, b, a, b, …`, then
`m, n, m, n, …`, and so on. A pair's stretch may be long or short, and a pair may come once or often. Nothing
follows a stretch but silence, so what a pass teaches is the alternation itself.

# 2. What the letters build

**The two relations.** `y` at frame 2 sees `x` one back: the relation `(x, y, 1)`. `x` at frame 3 sees `y`
one back: `(y, x, 1)`. Each neuron counts its own (D47), and from frame 4 on each relation has come round
twice (R21).

**The first function.** At frame 4, `(x, y, 1)` has strength 2 and no function holds it, so it starts one: two
nodes, `x` and `y`, one relation (R50). The fit is seen at frame 4, where it lands on `x` at 3 and `y` at 4. It
covers the relation and both names, 3, for a call of 1, and is accepted (R24). Its interneuron, `XY`, is born
at frame 3, the origin, one frame old, covering frames 3 and 4 (D53, D9).

**The cycle closes.** At frame 5, `x` fires, and its neuron returns `(y, x, 1)`, strength 2. That relation
shares `y` with an activation the standing call `XY` landed on, so it is a candidate of `XY`'s function,
attached at this placement (R50). `XY` is placed at every pair `x, y` in the stretch, and `x` follows every
one of them, so the candidate is attached at every accepted placement, and it joins: the function is now `x`
to `y` at 1 and `y` to `x` at 1, two nodes, two relations, a cycle (D15, R54). The relation a candidate
would add from `x` to `x` at 2 is implied by the other two along the cycle, and is never held (D34).

**One call over the run.** From then on a stretch of the pair is one call. Its fit is seen at the first `y`,
the call is born at the `x` before it, and every later letter is what the function expects: `x` after `y`,
`y` after `x`, each covered the frame it fires, never standing on the apex, and no interneuron fires anew (D54,
D10). The call goes round the cycle as many times as the stretch runs, and ends on the first relation that
does not come true, the silence after the last letter (D12).

| a stretch of `2k` letters | numbers |
|---|---|
| bare: `2k` names and `2k − 1` relations | `4k − 1` |
| the call: its name, how far round the cycle it ran, and one failure where the stretch ends | 3 |

The function costs 4 in the dictionary, two names and two offsets (D56, D13).

**Other pairs.** `a, b, a, b, …` builds `AB` the same way, and `m, n, m, n, …` builds `MN`. Each is a function of
two nodes and two relations, each has its own interneuron, and none of them shares a node with another.

# 3. The standing call

The call is on the apex over every frame it covers (D49). While it stands, its function says what comes next,
and that is placed at the frame ahead, weakly, as the call's own proposal (R30, R45): `y` after `x`, `x` after
`y`. When the letter arrives, the strong activation replaces the weak one and is covered. The call connects, at
each age, to what stands on the apex outside its function (R31): nothing, in this stream, until the silence.
What follows the stretch is what it learns, offset by how long the stretch was.

A letter covered at age 0 writes one connection, to the call, and is silent (D10). The call speaks for the
stretch: at the base, a letter's own connections say only that `x` is followed by `y` and `y` by `x`, which the
function says already, so after the first pass nothing is inferred at the base that the call does not say.

# 4. The count

The function's counts are over its accepted placements (R50, D57). Each pass is one placement, the two
relations come true once per traversal, and the failure at the end is the one the call ended on. Over a window
of passes each relation is worth `2 · held − failed − 2`, with `held` the traversals and `failed` the passes,
so a pair that comes round at all keeps its function, and one that stops coming sees its placements age out and
its function pruned (R18).

# 5. What is not settled

`XY`, `AB` and `MN` have the same shape: two nodes, two relations, offset 1 each way. Compared as graphs with
the names ignored, their nodes pair (R52), and at each node the two functions hold different neurons. If the
pairing were fixed, one general function would follow: two open nodes and the two relations, whatever the pair,
and a stretch of a pair never seen would be covered on its first pass, its two letters bound as values, for 3
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
