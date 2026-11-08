# A stream of letters

A worked case for [algorithm.md](algorithm.md): one channel of letters, one letter per frame, and the stretches
that come to be named from them. Nothing here is normative. It follows the definitions from the base, where a
letter sees only the letter before it, to the interneurons that stand for the stretches that recur.

---

# 1. The environment

One channel, `text`, laid out over time alone: a letter sees the letter before it and nothing further (D4). It
declares one event dimension, `letter`, with a bucket per letter and one for the space; each bucket is a base
neuron (D52). One frame is one letter. There is no action dimension: this machine reads.

A bare letter costs its relation to the letter before it, one relation (D12, D13). A call costs nothing of its
own: one per relation that did not come true and one per argument, and nothing else.

# 2. The base: one letter back

At the base the reach is 1 (D4), so a letter's neighborhood is one neighbor, the letter before it (D7). Its
activate call returns one relation per activation: the letter before, at `−1` in time, with its strength, how
many of the letter's activations saw it (D34, R21). The row that records the activation takes the letter after
it a frame later, when the connect call hands it what stood (D18, §7.1), so a letter's row holds the letter on
each side of it, and a relation between two letters is held by both of them.

A relation at strength 1 is returned and read in the index, to find a function that already fits it; nothing
is built on it (R21, R53). At strength 2 it is grouped into a function of two fixed nodes (R50). Take `th`.
Once `h` has seen `t` one back twice, `{h, t@−1}` is a function: `t` and `h` one frame apart, origin `h`
(D15). Its price:

| | relations |
|---|---|
| A, the relation between the two | 1 |
| B, the call's cost: nothing | 0 |
| savings per occasion | 1 |
| entry in the dictionary: one relation | 1 |
| worth at `n` occurrences | `n − 1` |

**A pair is taken at its second occurrence.** The fit covers one relation for nothing, and a fit is taken
while it covers more than it costs (R24), so at the second `th` the function's interneuron, call it `TH`, is
allocated (R16) and fires at `h`'s position, at age 0, in the pose from `h` back to `t` (D53, D37). It stands
for the two letters, which leave the apex, each connecting once more to it (D10), and inherits `t`'s relation
to the letter before it, now from the call at `−2`. Every pair of letters that recurs is such a function with
its interneuron, and leaves when its relation's strength slides below 2 (R18).

**No variable forms in this stream.** Two functions that differ at a node, `the` and `she` below, are
generalized into `{e, h@−1, C@−2}`, `C` a class whose members are `t` and `s`, and the general function is
priced as a replacement of the two (R52). The two cost 2 each in the dictionary and nothing per occurrence.
The class function costs 2 in the dictionary and one binding per occurrence, since the member that stood is an
argument (D13). It is never cheaper from the second occurrence on, and it does not join. The dictionary holds
one function per stretch that recurs, and nothing general.

# 3. Words

A word is not one neuron. What the machine names is the stretch that recurs, a letter at a time: a pair at its
second occurrence, and the longer stretch as its pairs merge.

Take `the`. At the second `the`, `TH` fires at the `h`, and the `e` after it sees the call one back (D5): the
count reads the call as the letters its expansion puts there, `h` at `−1` and `t` at `−2` (R50), so `e` after
`h` is at strength 2. It is no candidate of `TH`'s function, since it would make `e` later than that function's
origin `h` (R50), so it starts a function of its own, `{e, h@−1}`, origin `e`. Its fit lands `e` on the `e` and
`h` on `TH`'s call, whose pose starts at `h`, and covers the call whole (D54): one relation for nothing,
taken. Its interneuron, call it `HE`, fires at `e`'s position and stands for `e` and, through `TH`, for `t h`
(D53).

At the third `the` the relation `t` before `h` is a candidate of `HE`'s function at its past end: it shares the
node `h` with the fit, and `t` is earlier than the origin (R50). The node it attaches at counts it. `h`'s row
holds `t` at `−1` and `e` at `+1` (D18), so `h` counts how many of its activations hold `e` after it together
with `t` before it, and how many hold `e` after it without `t`: `with` and `without` (D47, R50). Over the rows
that have taken what followed them, the `h` of the first and second `the`, `with` is 2 and `without` 0, so the
candidate's worth in the function is `with − without − 1`, 1. Its contribution where it stands, alone in
`TH`'s function, read at the same node over the same rows, is its strength less one, 1. A tie, and a tie joins
where it leaves fewer functions (R54): the relation moves, `TH`'s function is empty and
goes with it (R18), and the function is `{e, h@−1, t@−2}`, three nodes, two relations, origin `e`, with `HE`
as its interneuron (R54).

A function that changed is fit again before the second election (R50). Its fit at the third `the` lands `e` on
the `e` and `h` on `TH`'s call, which stands for `t h`, and covers it whole: one relation for nothing, taken
(R24). `HE` fires once, at `e`'s position, at age 0, in the pose the fit read: from `e` back to `t`, one lap of
a function with no cycle (D53, D37). From the fourth `the` on nothing fires at the `h`, since `h` after `t` is
in no function of its own, and the fit at the `e` lands on three bare letters, two relations for nothing. The
call stands for the three letters and inherits the relation the fit did not cover, `t`'s relation to the letter
before it, now from the call at `−3` (D10). In the row of that earlier letter, which saw `t` at `+1`, the
machine changes the entry to `HE` in that pose at `+3`. Each of the three letters connects once more, to `HE`,
and is silent; `e`, covered at age 0, writes nothing else (D10). All three leave the apex (D49).

| `the`, priced as | relations |
|---|---|
| bare: three letters, two relations | 2 |
| through `HE`: nothing for the call | 0 |
| entry in the dictionary | 2 |
| worth at `n` occurrences | `2n − 2` |

Once a function is in the dictionary, it is found from a single occurrence and from any of its nodes (R53, D54):
one interneuron for every `the` in the stream (D11).

**The stretch grows at its past end.** `HE` sees one beyond what it landed on (D4), the letter before `t`.
Where that recurs, its pair with `t` is a function at its second occurrence, and its relation is a candidate at
`HE`'s function's past end, which `t` counts: its row holds the letter before it and, since the call replaced
`h` and `e` there, `HE` after it. After a space, the relation moves on the tie, and `{e, h@−1, t@−2, space@−3}`
is the function at the second ` the`, four nodes, three relations, covering 3 for nothing. So a stretch grows
one letter per recurrence toward its past, flat, in one function, and nothing has to stand on the board for it
but the letters.

**What follows is a function over the call.** The letter after `e` sees `HE` one back, the call standing at
`e`'s position (D5). Where a space has stood after `HE` twice, `{space, HE@−1}` is a function. Its fit lands on
the space and on the call, and covers the relation to the call and the call's cost, nothing (D22): 1 for
nothing, taken. Its interneuron stands for `t h e ` through `HE` and expands to it (R28). So a stretch is
reached two ways, flat at its past end and as a function over the call of what came before, and where both
fit the election takes the one that covers the most for its price (R24): the flat ` the` at 3 for nothing
before the pair over the call at 1 for nothing.

**One word, several interneurons.** Which stretch an interneuron stands for is decided by what recurred twice
first. After a space, the stretch grows to ` the`; where `n` stood before the space twice, to `n the`; after
`of`, to `of the`. These are different stretches, and the index gives one interneuron to a function whose
expansion comes round to what an interneuron already stands for, over exactly that length (D36, R43), and none
of these does. So `the` has one interneuron after a space, another in `n the`, another in `of the`. The
machine compresses the stretch that recurs, not the word.

A stretch of any length forms this way, and nothing has to be a power of two long: a letter is on the apex,
and the letter before it within reach can join the function that names it. A long stretch does not see far for
being long: an interneuron standing for eight letters sees one beyond them at each end, eight back from its
position and one further (D4).

# 4. A doubled letter

`ll`, `oo` and `ss` are pairs like any other: the second `l` sees the first one back, a neuron being its own
neighbor at any offset but zero (D26). `{l, l@−1}` has one node and one relation from it to itself, so its
entry is 1 and it is a cycle of one node (D15): `l` after `l` for as long as it runs. At a doubled letter its
fit reads one lap, 1 for nothing, and is taken: the call fires at the second `l`, in a pose of one lap (D53).
A longer run is covered a letter per frame: each new call reads one lap, lands on the call before, whose pose
starts at `l`, covers that call's cost, nothing, and the new relation, 1 for nothing, and its laps are the one
it read and the old call's together, so the run is one call however many have grown over it (R23). The same
holds for `oo` from `o`'s history and `ss` from `s`'s. A doubling too rare to recur within `l`'s history stays
as letters on the apex.

# 5. What the apex holds after a sentence

Interneurons for the stretches that recur, each standing at the last letter of its stretch in the pose it read
it in; the letters nothing covered; and the spaces, which are letters like any other until a function names
them. The apex is what the next letter sees (D5), and what connects to what follows (D25): every uncovered
activation open at the next frame connects to the letter that stood there (R31), a call under its pose like a
letter. It is also what votes (R27): each activation that fired at the present proposes at the frame ahead the
letters its connections name, and the letter the most voters name is output, weakly, for the report to
replace (R47, R40, D9).
