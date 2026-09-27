# An alternating pair, generalized

A worked case for [algorithm.md](algorithm.md): how `x, y, x, y`, `a, b, a, b` and `m, n, m, n` become one
function held once, by a neuron that keeps seeing a different pair in the same places. Nothing here is
normative. It follows one neuron's table through the greedy pick, the collapse of its residual, the price and
the election.

---

# 1. The stream

One event dimension, letters, laid out over time. The stream runs `x, y, x, y, p`, or `a, b, a, b, p`, or
`m, n, m, n, p`: an alternating pair, and then `p`. Three pairs recur; many others go by once. The pairs never
occur together. What they share is `p`, which follows every one of them, and `p` is the neuron this case
follows. What the letters build in their own tables is left out.

The levels are stated, not drawn: `p` is taken to stand at a level whose reach spans four frames (D4), and
offsets are written as plain frame counts.

# 2. The greedy pick: a pattern per recurring pair

Every activation of `p` recorded the four letters before it (D7). The greedy pick (D33) builds a pattern out of
whatever recurs, so each pair that came round twice gets one:

| in `p`'s table | the pattern | its child |
|---|---|---|
| for `x, y` | `(y, 1 back)`, `(x, 2 back)`, `(y, 3 back)`, `(x, 4 back)` | `P` |
| for `a, b` | `(b, 1 back)`, `(a, 2 back)`, `(b, 3 back)`, `(a, 4 back)` | `Q` |
| for `m, n` | `(n, 1 back)`, `(m, 2 back)`, `(n, 3 back)`, `(m, 4 back)` | `R` |

Each writes its occurrence as one symbol where the letters cost four. The pairs seen once stay in the residual,
written flat.

# 3. Parameters from what recurs without a name

Beside the three recurring pairs, many pairs have gone by once: `s, t, s, t, p`, `u, v, u, v, p`, and so on.
No letter recurs in them, so no pair of neurons reaches a count of two in the residual. What recurs is a
relation between offsets: one back and three back hold one neuron in every such row, and so do two back and
four back. The greedy pick seeds on the first of those relations (D33) and re-centers over the rows that hold
it (D27):

| Question | Answer |
|---|---|
| Does any neuron hold a majority at any offset? | No. No constants. |
| Which offsets hold one neuron in a majority of the rows? | One and three back; two and four back. |
| So what does the candidate name? | A parameter `P` over one and three back, and a parameter `Q` over two and four back. No class. |

```
(P, 1 back, 3 back),  (Q, 2 back, 4 back)
```

`P` says that one back and three back hold the same letter, and `Q` the same for two and four back (D38). The
pattern names no letter and keeps no list of letters.

# 4. The price

| written as | symbols |
|---|---|
| a once-seen pair, flat: `p` and four letters | 5 |
| a recurring pair, by its own pattern: `Q`'s child | 1 |
| a once-seen pair, by the general pattern: the instance, and a value child for each parameter | 1 + 1 + 1 = 3 |

The general pattern saves two on every once-seen pair and nothing on the recurring ones, which their own
patterns write for one. Its line costs three, one and a symbol each for `P` and `Q`, and each parameter costs
three in the parameters table (D13), so it pays once five or so rare pairs have gone by. `P`, `Q` and `R` stand
while their pairs are frequent; when one becomes rare its pattern retires (R18) and the general pattern writes
that pair too, with nothing to learn first.

# 5. The return and the election

`p` bids the general pattern whenever it is in a cover, carrying the letter each parameter held (D31). Nothing
has a neuron yet. The first time the bid is accepted the machine gives the pattern a child, `alternation`, and
each parameter a value child for the letter it passed, `P:t` and `Q:s`, one level above `p`, holding nothing
(R16, R44).

# 6. One occurrence

`s, t, s, t, p` arrives, a pair never seen, and `p` fires.

| Stage | what happens |
|---|---|
| `process functions`, in `p` | No specific pattern fits. The general one does: one neuron one back and three back, one neuron two and four back. `p` bids `alternation`, carrying `P = t` and `Q = s`. |
| the election | The bid covers `p` and the four letters for a price of three, and is accepted. |
| activate children | One level up, `alternation` fires at `p`'s coordinate, `P:t` at the nearest `t` and `Q:s` at the nearest `s` (§7.4). |

The level above reads `alternation` beside two value children, `P:t` and `Q:s`: which function ran, and with what.
Nothing about `s` or `t` had to be learned first; only their value children are new.

`s, t, u, t, p` does not fit fully: `P` holds `t` at both its offsets, but `Q` holds `s` at two back and `u` at
four back. `Q`'s value is the nearer, `s`, and four back is a correction (D44, D22).

# 7. What the one child buys

`alternation` is one neuron whatever the pair was, so its connections are one set (D25): what the machine learns
to do after an alternation it has learned for every pair, and the value children standing beside it say which
pair this was. Each value child is a neuron too, `P:t` with a history of everything that followed an
alternation whose first letter was `t`, so the level above can learn about a particular pair where that matters,
by a pattern naming `alternation` and `P:t` together.

# 8. The same thing as code

| in the code | in the machine |
|---|---|
| three copies of one block with different names in them | `P`, `Q` and `R` in `p`'s table |
| noticing they are one block with two names that vary | the collapse finding four offsets that agree in pairs |
| the parameters | `P` and `Q`, in `p`'s parameters table |
| the extracted function | the general pattern, held once in `p`'s table |
| a call with its arguments | `alternation` firing, with `P:t` and `Q:s` beside it |
| keeping the hot path inlined | `Q` staying in the table while `a, b` is frequent, and a pattern one level up that names the call with a value child |
