# An alternating pair, generalized

A worked case for [algorithm.md](algorithm.md): how `x, y, x, y`, `a, b, a, b` and `m, n, m, n` are seen by a
neuron that keeps seeing a different pair in the same places, and what it comes to hold for them. Two things,
in one neuron's history: a general function, with a parameter tying the letters two apart, twice over,
generalized from two pairs that each came round twice and fitting any pair after that; and, for each pair that
comes often, the interneuron its own letters build, named outright. Nothing here is normative. It follows the
letters' own functions first, then `p`'s, through grouping, finding, generalizing, the price and the election.

A function is written here as its nodes, each at its offset in time from the node in its latest frame. The
dictionary holds it as relations, each written from the older of its two nodes (D15, D34).

---

# 1. The stream

One channel, `letters`, laid out over time alone, with radius `R = 4` (D1): a letter sees four frames back, and
every distance within that is exact (D6). The stream runs `x, y, x, y, p`, or `a, b, a, b, p`, or
`m, n, m, n, p`: an alternating pair, and then `p`. Three pairs recur; many others go by once or twice. The pairs
never occur together, and between passes the stream is silent for longer than any reach, so a pass sees nothing
of the one before (D4). What the passes share is `p`, which follows every one of them, and `p` is the neuron
this case follows.

Every base neuron here has reach `4` (D4). `p` sees the four letters before it at `−1`, `−2`, `−3`
and `−4`, each at its own offset, which is what the radius buys: at `R = 1` the letters two and three back would
both be written at `−2` (D6), and nothing could say that one back and three back held the same letter.

# 2. What the letters build

The letters fire before `p` in the stream, and what they build decides what `p` sees, since a neighbor is
whatever stands on the apex, and an interneuron there is a neighbor as itself and as each letter under it
(D5, D47).

**A pair's first pass names its last two letters.** In `s, t, s, t`, `t` at frame 2 sees `s` at `−1`, and `t`
at frame 4 sees `s` at `−1`, `t` at `−2` and `s` at `−3`. `(t, s, −1)` is held in two rows, so grouping makes
`{t, s@−1}` (R50): a row written bare is `4`, through the function `2`, which saves `4` over two rows against
an entry of `3`, margin `1` (D30). It is new (R53). Elected at frame 4, it covers `s` at frame 3 and `t` at
frame 4, `4` for `2` (R24), and its interneuron, `ST`, fires at frame 4 (D53). It was placed from the `t` at
frame 4 (D31), so the `s` and `t` at frames 1 and 2 stand bare.

**The second pass covers the four letters whole.** At frame 12 the second `t` finds `{t, s@−1}` as it is,
and `ST` fires over frames 11 and 12. The `s` at frame 13 sees `ST` at `−1`, and under it `t` at `−1` and `s`
at `−2`. `(s, t, −1)` and `(s, s, −2)` are each held in two rows, frames 3 and 13, and group into
`{s, t@−1, s@−2}`. Row 3 is `6` bare, `2` through it. In row 13 it lands on both letters under `ST`, so it is
`s` and `ST`'s record, `4`, for `2` (D22, R23). Savings `6`, entry `4`, margin `2`. Elected, it takes `ST`'s
call back, and `STS` fires at frame 13. The `t` at frame 14 sees `STS` at `−1` and the three letters under
it, and `t` at `−2` and `s` at `−3` are each held in two rows, frames 4 and 14. `{t, s@−1, t@−2, s@−3}`
writes row 4 at `2` where `ST` and two bare letters wrote it at `6`, and row 14 at `2` where `t` and
`STS`'s record wrote it at `4`: savings `6`, entry `5`, margin `1`. Elected, it takes `STS`'s call back, and
`STST` fires at frame 14. Every later pass runs the same chain, `ST`, `STS`, `STST`, each taking the one
before it back, and what `p` sees one back is one interneuron standing for the four letters. `xyxy` and
`mnmn` form the same way, each in its own letters' histories.

**A pair that goes by once builds `ST` and nothing more.** Its first pass is the first pass above: one function
over its last two letters, and the first two bare. Those are the passes the general function is elected on.

# 3. What `p` sees

Every activation of `p` records what stood within four frames (D7). On a pair's first pass that is `ST` at
`−1`, with `t` at `−1` and `s` at `−2` under it, then `t` at `−3` and `s` at `−4`, bare. On a later pass it is
`STST` at `−1`, with the four letters under it at `−1` through `−4`.

On the second pass of `s, t` the relations `(p, t, −1)`, `(p, s, −2)`, `(p, t, −3)` and `(p, s, −4)` are each
held in two rows, frames 5 and 15, and group into `{p, t@−1, s@−2, t@−3, s@−4}`. Row 5 is `p`, `ST`'s
record and two bare letters, `8`, for `2`: the function lands on both letters under `ST` and takes its call
back. Row 15 is `p` and `STST`'s record, `4`, for `2`. Savings `8`, entry `7`, three neurons and four
relations, margin `1`. It is new. Elected at frame 15, it covers `4` for `2`, takes `STST`'s call back, and
its interneuron fires at frame 15.

# 4. The function

The second pass of a second pair, `x, y`, groups `{p, y@−1, x@−2, y@−3, x@−4}` the same way. Find reads each
relation with either end open (R53): `(p, ?, −1)` reaches the function over `s, t`, which writes `(p, t, −1)`,
and so do `(p, ?, −2)`, `(p, ?, −3)` and `(p, ?, −4)`. It is in every relation's reading, only through an open
end, so the two differ only at the open ends, and the machine generalizes them (R52). `p` is the same in
both, so it is fixed. The nodes at `−1` and `−3` differ, and within each function hold one neuron, `t` in one
and `y` in the other: one parameter, `P`. The nodes at `−2` and `−4` differ the same way: a second parameter,
`P′`. Nothing is left to fix:

| Question | Answer |
|---|---|
| Is anything left for the function to name as a fixed node? | No. The two variables hold all four places. |
| What holds in those rows? | `p` itself, one letter at `−1` and `−3`, and one letter at `−2` and `−4`. |
| So what does the function name? | `p`, `P` at `−1` and `−3`, and `P′` at `−2` and `−4`, and nothing else. |

```
alternation:  p;  P at −1 and −3;  P′ at −2 and −4
```

The function fixes no letter. It says that two letters alternated before `p`, whatever the pair was. Its one
fixed node is `p`, in its latest frame (D15). Its entry is `p` and four relations, `5`, and its call's record
is `2` and two values, `4`. It is priced as a replacement over `p`'s four rows (R52): the two specifics cost
`14` in the dictionary and `2` per row, `22`; `alternation` costs `5` and `4` per row, `21`. Cheaper by `1`,
it joins, and both specifics leave the dictionary. Each variable's members are the letters that have stood at
both of its places, a record of what has passed and no condition on what may (D44).

At that same frame it is placed and measured: `p` and `xyxy`'s record, `4`, for `4`. Coverage does not exceed
price, so it is not taken (D28), and `p` and `xyxy` stand on the apex. It has no interneuron until it is first
accepted (R16).

# 5. The recurring pairs

On the third pass of `x, y`, `p` sees `xyxy` at `−1`, and `(p, xyxy, −1)` is held in two rows. `{p, xyxy@−1}`
writes each at `2` where `p` and `xyxy`'s record wrote it at `4`: savings `4`, entry `3`, margin `1`. Find's
open reading `(p, ?, −1)` reaches `alternation`, but the two do not differ only in which neuron stands at some
of their nodes, since `alternation` has nodes this function lacks, so generalizing makes nothing (R52). The
function is new. Elected, it covers `4` for `2`, where `alternation` measures `4` for `4`, and its interneuron
fires at `p`'s frame. `stst` and `mnmn` get a function each the same way.

A class over the three is made when the second of these functions is found through the first's open end
(R52), and never pays. `m` functions of one chain each cost `3m` in the dictionary and `2` per pass; one
function over a class costs `3 + m` and `3` per pass. The class is cheaper only while the passes number fewer
than `2 − 3/m` per chain, and each function already has two.

So the two forms stand in one history and never meet in one row. What comes often is covered by its own
letters' chain and named here outright; what comes rarely is described by the general function. A pair that
becomes frequent moves from the one to the other as its letters' chain covers it, and nothing in `p`'s
dictionary is told: its rows stop holding four letters and start holding one interneuron.

# 6. The price

The stream is laid out over time alone, so a bare activation's record is `2` (D13). A call's record is `2`,
one per value, and one per relation that did not come true (D12).

| a pass, priced as | numbers |
|---|---|
| flat: `p` and four letters, each at what it cost | 10 |
| a first pass, as it stands on the board: `p`, the pair's last two letters under their call, two bare letters | 8 |
| by `alternation`: its record and two values | 4 |
| a later pass, as it stands on the board: `p` and the letters' chain | 4 |
| by `{p, stst@−1}`: its record | 2 |

`alternation`'s entry is `5`: one number per relation, four, and `p`. The variables' members cost nothing (D44).
A letter new to a variable is its value the first time it passes, and a member from then on.

What `alternation`'s call fires stands on the frontier at what it costs: its record, `4`, at `p`'s frame. The
four letters stay bound, their positions covered, each written once as its parameter's value (D10). Whatever
covers them later is credited their neurons and no more (D24).

# 7. The return and the election

On a pass `p` has not seen, its activate call returns its relations held in one row (R21). The machine reads
them in the index the three ways, and the readings with the letter end open reach `alternation` (R53).
`alternation` fits where one letter stands at `−1` and `−3` and one at `−2` and `−4` (D54). It is placed
with the letter in each variable (D31) and elected where it covers more than it costs (R24). The first time
it is accepted the machine allocates its interneuron, `alternation`, one level above `p`, holding nothing
(R16).

# 8. One occurrence

`s, t, s, t, p` arrives, a pair never seen, and `p` fires. The letters built `ST` over frames 3 and 4 (§2),
and the first two stand bare.

| Stage | what happens |
|---|---|
| activate, in `p` | `p` returns `(p, t, −1)`, `(p, s, −2)`, `(p, t, −3)`, `(p, s, −4)` and `(p, ST, −1)`, each held in one row. |
| find | The open readings reach `alternation`. `t` stands at `−1` and `−3` and `s` at `−2` and `−4`, so it fits, with `P = t` and `P′ = s`. |
| elect | It covers `p`, the two bare letters and `ST`'s record, `8`, for `4`, and is taken. `ST`'s call is taken back. |
| allocate | `alternation` fires at `p`'s frame; the four letters stay where they are, bound (§6.4). |

The level above sees the call neuron and, bound beside it, the four letters where they stood. Nothing about
`s` or `t` had to be learned first: each joined its variable's members in this row.

`s, t, u, t, p` fits with one error. `ST` covers frames 1 and 2, and the `t` at frame 4 builds nothing over
`u`. `P` holds `t` at `−1` and `−3`. `P′` finds `u` at `−2` and `s` at `−4`: its value is the neuron at the
earliest of its nodes, `s`, and the relation at `−2` did not come true (D44). It covers `p`, the `t` at
frame 4 and `ST`'s record, `6`, for `2`, two values and one error, `5`, and is taken; the `u` stands bare.

`x, y, x, y, p` arrives on a later pass, and `p` sees `xyxy` at `−1` and the four letters under it (§2).
`alternation` lands on all four and measures `4` for `4`; `{p, xyxy@−1}` measures `4` for `2` and is taken.
Its interneuron fires at `p`'s frame, covering `p` and `xyxy`.

# 9. What the one interneuron holds

`alternation` is one neuron whatever the pair was, so its connections are one set (D48): what the machine
learns to do after an alternation it has learned for every pair it saw on a first pass, a pair never seen
before included. The letters bound beside it say which pair this was, and the level above can learn about a
particular pair where that matters, by a function over `alternation` and the bound `t` together, whose
interneuron then holds what followed an alternation of `t`.

A frequent pair is particular already. `{p, xyxy@−1}`'s interneuron holds what followed that pair and nothing
else. What `xyxy` has in common with `abab`, that each is two of one pair, is said by `alternation`, which fits
both; but it is not taken over either, since each pair's own chain writes its letters shorter (D28). That is
the trade the design makes everywhere: what recurs is named, and the general function is elected where
nothing more specific is.

# 10. The same thing as code

| in the code | in the machine |
|---|---|
| three copies of one block with different names in them | `x, y, x, y`, `a, b, a, b` and `m, n, m, n` in the stream |
| noticing they are one block with two names that vary | two functions in `p`'s history that differ only at the letters, generalized (R52) |
| the parameters | `P` and `P′`, two parameters of one function |
| the extracted function | `alternation`, a function over `p` and the two variables, held once in the dictionary |
| a call with its arguments | `alternation` fires, with `t` and `s` bound beside it |
| keeping the hot path inlined | `xyxy`, one interneuron for a frequent pair, and `p`'s function naming it |
