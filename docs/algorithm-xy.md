# An alternating pair, generalized

A worked case for [algorithm.md](algorithm.md): how `x, y, x, y`, `a, b, a, b` and `m, n, m, n` are seen by a
neuron that keeps seeing a different pair in the same places, and what it comes to hold for them. Two things, in
one table: a general form, two parameters and one function, learned from the pairs that go by once or twice; and
a memorized chunk, named by a class, for each pair that comes often. Nothing here is normative. It follows the
letters' own tables first, then one neuron's table through the greedy pick, the collapse of its residual, the
price and the election.

---

# 1. The stream

One channel, `letters`, laid out over time alone, with radius `R = 4` (D1): a letter sees four frames back, and
every distance within that is exact (D6). The stream runs `x, y, x, y, p`, or `a, b, a, b, p`, or
`m, n, m, n, p`: an alternating pair, and then `p`. Three pairs recur; many others go by once or twice. The pairs
never occur together. What they share is `p`, which follows every one of them, and `p` is the neuron this case
follows.

Every neuron here is a base letter with reach `4` (D4). `p` sees the four letters before it at `−1`, `−2`, `−3`
and `−4`, each at its own offset, which is what the radius buys: at `R = 1` the letters two and three back would
both be written at `−2` (D6), and nothing could say that one back and three back held the same letter.

# 2. What the letters build

The letters are called in the same level as `p` and before it in the stream, and what they build decides what
`p` sees, since a neighbor is whatever stands uncovered on the frontier (D5).

**A pair that comes often is chunked.** `y` keeps seeing `x` one back, so its table holds a function naming `x`
at `−1`, with whatever else recurs around it (D47, D27), and its child, `xy`, fires at `y`'s frame covering both
letters. The second `x` of the alternation then sees `xy` one back, not `y`, and comes to name it in turn; the
second `y` sees that child, and names it. So a frequent alternation is chunked whole, a letter at a time, into
one neuron, `xyxy`, standing at the frame of the last `y`, as a word is ([algorithm-text.md](algorithm-text.md)).
`abab` and `mnmn` form the same way, each in its own letters' tables. What `p` then sees one back is one neuron,
and the letters under it are covered and not in its neighborhood.

**A pair that goes by once or twice builds nothing.** A pattern needs a relation counted at least twice in its
owner's history (D33) and a margin that pays its line (R15). `s, t, s, t` gives `t` two rows with `s` one back,
and a line that names a neuron among every neuron the machine holds costs more than two occurrences of `s` save,
so nothing is built, and `s, t, s, t` stands on the frontier as four letters. Those are the rows the general form
is learned from: `p` sees them flat, at `−1` through `−4`.

# 3. Variables first: two parameters

Every activation of `p` records what stood within four frames (D7): for a chunked alternation, one neuron at
`−1`; for a bare one, four letters. The greedy pick builds variables first (D33), and the bare rows are where it
finds them. In every one of them, whatever the pair, one back and three back hold one neuron, and so do two back
and four back. A pair of offsets that agree is a relation a variable is seeded on (D47), and each is grown by its
own collapse and kept on its own uses (D27). Positions that hold one and the same neuron are a parameter (D44):

```
P:  (−1, −3)        Q:  (−2, −4)
```

`P` says that one back and three back hold the same letter, and `Q` the same for two and four back (D38). Each
writes two letters as one member, and its members are the letters that have stood at both its offsets, a record
of what has passed and no condition on what may; neither keeps a list of the pairs. They form as soon as two bare
alternations have come round, the same pair or two different ones.

# 4. The function over them

Then the pick builds functions, over the rows as the variables have rewritten them. `P` and `Q` hold together in
every bare row, which is a relation between two variables, and the collapse grows it into a function (D27):

| Question | Answer |
|---|---|
| Is anything left for the function to name as a neighbor? | No. `P` and `Q` hold all four offsets, and a position a variable holds is that variable's (D27). |
| Which variables hold in those rows? | `P`, over one and three back; `Q`, over two and four back. |
| So what does the function name? | The two parameters, and nothing else. |

```
alternation:  P, Q
```

The function fixes no letter. It says that `P` and `Q` held together, which is the alternation, whatever the
pair was.

# 5. The recurring pairs

The three pairs that recur are not written by `alternation`. By the time `p` fires they are one neuron each,
`xyxy`, `abab` or `mnmn`, standing one back (§2), and the letters under them are covered (D5). One offset where
different neurons stand is what a class is seeded on (D47), so `p`'s table holds a class at `−1` whose members
are the alternations that have recurred there (D41), and a value neuron for each, `p:xyxy` (D45). A class is
closed on its members: a chunk that has not stood there before is a failed neighbor until it has recurred (D27).

So the two forms stand in one table and never meet in one row. What comes often is memorized as a chunk by its
own letters and named here by the class; what comes rarely is described by the general form. A pair that becomes
frequent moves from the one to the other as its letters chunk it, and nothing in `p`'s table is told: its rows
simply stop holding four letters and start holding one neuron.

# 6. The price

Say the base alphabet has thirty-two symbols, so a letter costs 5 bits to write, and say `p` cost 5 as well.
`P` and `Q` each hold eight letters as members, so which member costs 3; and `p` holds four patterns, `P`, `Q`,
`alternation` and the class, so which of them costs 2 (D13).

| written as | bits |
|---|---|
| a pair, flat: `p` and four letters, each at what it cost | 5 × 5 = 25 |
| by `P` alone, the other two letters flat: `p`, which pattern, one member, two letters | 5 + 2 + 3 + 5 + 5 = 20 |
| by `alternation`: `p`, which pattern, and two members | 5 + 2 + 3 + 3 = 13 |

A second parameter bidding alone would write `p` and which pattern again, 10 bits, for two letters that cost 10,
so it does not pay (D28). Without the function a pair costs 20 bits, and `alternation` saves 7 on every one
(D30). Its line is two references, which pattern each, 4 bits, so it pays at its first occurrence. A parameter's
line is an offset per position and a neuron per member, and naming a neuron in a line is a choice among every
neuron the machine holds: at `R = 4` and a reach of 4 an offset in time is one of five, zero and the four exact
distances, about 2.3 bits (D6), so with two hundred and fifty-six neurons the line is `2 · 2.3 + 8 · 8 ≈ 69`
bits, paid once. A letter new to `P` is its value the first time it passes, since agreeing at both offsets is
the whole test (D27), and is a member from then on.

What the call fires stands on the frontier at what it cost to write: `alternation` at 7 bits, the owner and
which pattern, and `P:t` and `Q:s` at 3 bits each. Whatever covers them later saves that and no more (D13).

# 7. The return and the election

`p` sends `alternation` whenever it is in a cover, as one bid with `P` and `Q` and the letter each held (D31).
Nothing has a neuron yet. The first time the bid is accepted the machine gives the function a child,
`alternation`, and each parameter a value neuron for the letter it passed, `P:t` and `Q:s`, one level above
`p`, holding nothing (R16, R44).

# 8. One occurrence

`s, t, s, t, p` arrives, a pair never seen, and `p` fires. The letters built nothing (§2), so the four stand bare.

| Stage | what happens |
|---|---|
| `process functions`, in `p` | `P` holds `t` at one and three back and `Q` holds `s` at two and four, so `alternation` holds. `p` sends one bid: the function, with `P = t` and `Q = s`. |
| the election | The bid covers `p` and the four letters, 25 bits, for a price of 13, and is accepted. |
| activate children | One level up, at `p`'s coordinate: `alternation`, `P:t` and `Q:s` (§7.4). |

The level above reads three neurons at one coordinate: which function ran, and what it was given. Nothing about
`s` or `t` had to be learned first: each joined its parameter in this row, and only their value neurons are new.

`s, t, u, t, p` does not fit fully: `P` holds `t` at both its offsets, but `Q` holds `s` at two back and `u` at
four back. `Q`'s value is the nearer, `s`, and four back is a failed neighbor (D44, D22).

`x, y, x, y, p` arrives and `p` sees `xyxy` one back and nothing else (§2). `alternation` does not hold, since
nothing stands at its parameters' positions; the class holds, `p` sends it alone, and its value neuron
`p:xyxy` fires one level up.

# 9. What the one child buys

`alternation` is one neuron whatever the pair was, so its connections are one set (D25): what the machine learns
to do after an alternation it has learned for every pair it saw flat, a pair never seen before included. The
value neurons standing beside it say which pair this was. Each has a history and connections of its own, `P:t` of
everything that followed an alternation whose first letter was `t`, so the level above can learn about a
particular pair where that matters, by a function over `alternation` and `P:t` together.

A frequent pair is particular already. `p:xyxy` holds what followed that pair and nothing else, and what `xyxy`
has in common with `abab`, that each is two of one pair, is stated by no neuron here: the shape is in two lines
in two tables, and nothing reads one table against another. That is the trade the design makes everywhere.
What recurs is named, and the general form is learned from what does not
([algorithm-evaluation.md](algorithm-evaluation.md)).

# 10. The same thing as code

| in the code | in the machine |
|---|---|
| three copies of one block with different names in them | `x, y, x, y`, `a, b, a, b` and `m, n, m, n` in the stream |
| noticing they are one block with two names that vary | four offsets that agree in pairs: the two parameter relations |
| the parameters | `P` and `Q`, in `p`'s table |
| the extracted function | `alternation`, a function that names `P` and `Q`, held once in `p`'s table |
| a call with its arguments | `alternation` firing, with `P:t` and `Q:s` beside it |
| keeping the hot path inlined | `xyxy`, one neuron for a frequent pair, and `p`'s class over such neurons |
