# An alternating pair, generalized

A worked case for [algorithm.md](algorithm.md): how `x, y, x, y`, `a, b, a, b` and `m, n, m, n` are seen by a
neuron that keeps seeing a different pair in the same places, and what it comes to hold for them. Two things,
in one table: a general form, one function with a parameter tying the letters two apart, twice over,
learned from the pairs that go by once or twice; and a memorized pair, bound in a class, for each pair
that comes often. Nothing here is normative. It follows the
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

**A pair that comes often is named outright.** `y` keeps seeing `x` one back, so its table holds a function
naming `y` and `x` at `−1` from it, with whatever else recurs around them (D47, D27), and its child, `xy`, fires at `y`'s frame covering both
letters. The second `x` of the alternation then sees `xy` one back, not `y`, and comes to name it in turn; the
second `y` sees that child, and names it. So a frequent alternation is covered whole, a letter at a time, into
one neuron, `xyxy`, standing at the frame of the last `y`, as a word is ([algorithm-text.md](algorithm-text.md)).
`abab` and `mnmn` form the same way, each in its own letters' tables. What `p` then sees one back is one neuron,
and the letters under it are covered and not in its neighborhood.

**A pair that goes by once or twice builds nothing.** A pattern needs a relation counted at least twice in its
owner's history (D33) and a margin that pays its line (R15). `s, t, s, t` gives `t` two rows with `s` one back,
and a line that names a neuron among every neuron the machine holds costs more than two occurrences of `s` save,
so nothing is built, and `s, t, s, t` stands on the frontier as four letters. Those are the rows the general form
is learned from: `p` sees them flat, at `−1` through `−4`.

# 3. The same-to-same relation

Every activation of `p` records what stood within four frames (D7): for an alternation its letters have
covered, one neuron at `−1`; for a bare one, four letters. The bare rows are where the general form is found.
In every one of them, whatever the pair, one back and three back hold one neuron, and so do two back and four
back: two same-to-same relations at distance 2 (D34, D47), each holding in every bare row. Either is counted
twice as soon as two bare alternations have come round, the same pair or two different ones, and the pick
grows a function from it (D33).

# 4. The function

The collapse grows the relation into a function over `p` (D27). It ties one back and three back in a parameter
variable, `P`. At two back and four back, the other same-to-same relation, it ties a second parameter, `P′`,
since that covers the two letters there for one member in every use, where a fixed node would fail in most of
them. Nothing is left to name as a fixed node:

| Question | Answer |
|---|---|
| Is anything left for the function to name as a fixed node? | No. The two variables hold all four places. |
| What holds in those rows? | `p` itself, and the same letter at `−1` and `−3`, and the same letter at `−2` and `−4`. |
| So what does the function name? | `p`, `P` tying `−1` to `−3`, and `P′` tying `−2` to `−4`, and nothing else. |

```
alternation:  p;  P: −1 same as −3;  P′: −2 same as −4
```

The function fixes no letter. It says that two letters alternated before `p`, whatever the pair was. It names
`p`, so it holds at `p` alone and covers it (D38). Each variable's members are the letters that have stood at both
of its places, a record of what has passed and no condition on what may (D44).

# 5. The recurring pairs

The three pairs that recur are not covered by `alternation`. By the time `p` fires they are one neuron each,
`xyxy`, `abab` or `mnmn`, standing one back (§2), and the letters under them are covered (D5). An offset from
`p` where different neurons stand is a node-to-open relation (D34, D47), so `p`'s table holds a function naming
`p` and a class at `−1` from it, whose members are the alternations that have recurred there (D41). Where
one stands, it is bound beside the function's child (D10). A class is closed on its members: a child that
has not stood there before is a failed node until it has recurred (D27).

So the two forms stand in one table and never meet in one row. What comes often is memorized as children by its
own letters and named here by the class; what comes rarely is described by the general form. A pair that becomes
frequent moves from the one to the other as its letters' patterns cover it, and nothing in `p`'s table is told: its rows
simply stop holding four letters and start holding one neuron.

# 6. The price

Say the machine holds thirty-two neurons, so a letter costs 5 bits, and so does `p`.
`P` and `P′` each hold sixteen letters as members, every letter that has stood two apart, so which member
costs 4; and `p` holds two functions, `alternation` and the one over the class, so which of them costs 1
(D13).

| priced as | bits |
|---|---|
| a pair, flat: `p` and four letters, each at what it cost | 5 × 5 = 25 |
| by `alternation`: `p`, which function, and two members | 5 + 1 + 4 + 4 = 14 |

Without the function a pair costs 25 bits, and `alternation` prices it at 14, saving 11 on every one. Its
entry in the table is two parameters, an offset each: at `R = 4` and a reach of 4 an offset in time is one
of five, zero and the four exact distances, about 2.3 bits (D6), so under 5 bits, paid once, and it pays at its
second occurrence. The variables' members are a record and cost nothing in the table (D13). A letter new to a variable
is its value the first time it passes, since agreeing at both places is the whole test (D27), and is a member
from then on.

What the call fires stands on the frontier at what it costs: `alternation` at 6 bits, the owner and
which function, at `p`'s frame; the four letters stay bound, the first of each pair at 4 bits, which member,
and the second at nothing. Whatever covers them later saves that and no more (D13).

# 7. The return and the election

`p` sends `alternation` whenever it fits, as one bid with the letter in each variable (D31). It has no child yet.
The first time the bid is accepted the machine gives the function a child, `alternation`, one level above `p`,
holding nothing (R16).

# 8. One occurrence

`s, t, s, t, p` arrives, a pair never seen, and `p` fires. The letters built nothing (§2), so the four stand bare.

| Stage | what happens |
|---|---|
| `process functions`, in `p` | `t` stands at one and three back and `s` at two and four, so both variables hold and `alternation` fits. `p` sends one bid: the function, with `P = t` and `P′ = s`. |
| the election | The bid covers `p` and binds the four letters, 25 bits, for a price of 14, and is accepted. |
| activate children | One level up: `alternation` at `p`'s coordinate; the four letters stay where they are, bound (§6.4). |

The level above reads the call neuron and, bound beside it, the four letters where they stood: which function
ran, and what it was given. Nothing about `s` or `t` had to be learned first: each joined its variable's members in
this row.

`s, t, u, t, p` does not fit fully: `P` holds `t` at both its places, but `P′` holds `s` at two back and `u` at
four back. Its value there is the letter at the later place, `s`, and four back is a failed node (D44, D22).

`x, y, x, y, p` arrives and `p` sees `xyxy` one back and nothing else (§2). `alternation` does not hold, since
nothing stands at its variables' places; the function over the class holds, `p` sends it, its child fires at
`p`'s frame and `xyxy` stays bound one frame back, where it stood.

# 9. What the one child buys

`alternation` is one neuron whatever the pair was, so its connections are one set (D25): what the machine
learns to do after an alternation it has learned for every pair it saw flat, a pair never seen before included.
The letters bound beside it say which pair this was, and the level above can learn about a particular pair
where that matters, by a function over `alternation` and the bound `t` together, whose child then holds what
followed an alternation of `t`.

A frequent pair is particular already. `xyxy` holds what followed that pair and nothing else, and what `xyxy`
has in common with `abab`, that each is two of one pair, is stated by no neuron here: the shape is in two
functions in two tables, and nothing reads one table against another. That is the trade the design makes everywhere.
What recurs is named, and the general form is learned from what does not
([algorithm-evaluation.md](algorithm-evaluation.md)).

# 10. The same thing as code

| in the code | in the machine |
|---|---|
| three copies of one block with different names in them | `x, y, x, y`, `a, b, a, b` and `m, n, m, n` in the stream |
| noticing they are one block with two names that vary | two frames two apart that agree, twice over: two same-to-same relations |
| the parameters | `P` and `P′`, two parameters of one function in `p`'s table |
| the extracted function | `alternation`, a function over `p` and the two variables, held once in `p`'s table |
| a call with its arguments | `alternation` firing, with `t` and `s` bound behind it |
| keeping the hot path inlined | `xyxy`, one neuron for a frequent pair, and `p`'s class over such neurons |
