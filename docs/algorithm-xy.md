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

# 3. A class from what recurs without a name

Beside the three recurring pairs, many pairs have gone by once: `s, t, s, t, p`, `u, v, u, v, p`, and so on.
No letter recurs in them, so no pair of neurons reaches a count of two in the residual. What recurs is a
relation between offsets: one back and three back hold one neuron in every such row, and so do two back and
four back. The greedy pick seeds on the first of those ties (D33) and re-centers over the rows that hold it
(D27):

| Question | Answer |
|---|---|
| Does any neuron hold a majority at any offset? | No. No constants. |
| Which offsets hold one neuron in a majority of the rows? | One and three back; two and four back. |
| So what does the candidate name? | A class `K` under one mark at one and three back, and a class `J` under one mark at two and four back. |

```
(K¹, 1 back),  (J¹, 2 back),  (K¹, 3 back),  (J¹, 4 back)
```

`K¹` twice means one member at both; `J¹` likewise (D38). The pattern names no letter. `K`'s first members are
the letters that stood one and three back in those rows, `J`'s the others.

# 4. The price

`p` has seen some thirty letters beside it, and eight of them are members of `K`. A `K` activation therefore
costs `log₂ 8 / log₂ 30`, about six tenths of a symbol (D13); `J` the same.

| written as | symbols |
|---|---|
| a once-seen pair, flat | 4 |
| a recurring pair, by its own pattern: `Q` | 1 |
| a once-seen pair, by the general pattern: the instance, one `K` activation, one `J` activation | 1 + 0.6 + 0.6 = 2.2 |

The general pattern saves about 1.8 on every once-seen pair and nothing on the recurring ones, which their own
patterns write for one. Its line costs five, and each class its name and eight members (D13), so it pays once a
dozen or so rare pairs have gone by. `P`, `Q` and `R` stand while their pairs are frequent; when one becomes
rare its pattern retires (R18) and the general pattern writes that pair too, its letters joining `K` and `J` as
they stand there twice (D27).

# 5. The return and the election

`p` bids the general pattern whenever it is in a cover, with a class bid for `K¹` and one for `J¹`, each naming
the letter it was fit by (D31). Nothing has a neuron yet. The first time the bid is accepted the machine gives
the pattern a child, `alternation`, and each class a class neuron; no other bid of that election covers the same
ground or reads the same letters, so it creates all three, one level above `p`, holding nothing (R16, R41, R43).

# 6. One occurrence

`s, t, s, t, p` arrives, a pair never seen, and `p` fires.

| Stage | what happens |
|---|---|
| `process functions`, in `p` | No specific pattern fits. The general one does: one neuron one back and three back, one neuron two and four back. `p` bids `alternation`, with class bids for `K¹` at the nearest `t` and `J¹` at the nearest `s`. |
| the election | The bid covers `p` and the four letters for a price of three, and is accepted. |
| activate children | One level up, `alternation` fires at `p`'s coordinate; a `K` variable fires at the `t`, holding `t`; a `J` variable fires at the `s`, holding `s` (§7.4). |

The level above reads `alternation` beside two variables, `K = t` and `J = s`. Nothing about `s` or `t` had to
be learned first.

`s, t, u, t, p` does not fit: `K¹` is the same value at both its offsets, but `J¹` would have to hold `s` and
`u` at once. Both of `J¹`'s offsets count as named and absent, and the bid is priced accordingly (D22).

# 7. What the one child buys

`alternation` is one neuron whatever the pair was, so its connections are one set (D25): what the machine learns
to do after an alternation it has learned for every pair, and the variables standing beside it say which pair
this was. `K` and `J` are neurons too. Each has a history pooling the neighborhoods around every letter it has
held, and connections of its own, and uncovered it votes (D41).

# 8. The same thing as code

| in the code | in the machine |
|---|---|
| three copies of one block with different names in them | `P`, `Q` and `R` in `p`'s table |
| noticing they are one block with two names that vary | the collapse finding four offsets that agree in pairs |
| the variables | the class neurons `K` and `J` |
| the extracted function | the general pattern, held once in `p`'s table |
| a call, with the variables it reads | `alternation` firing, with `K` and `J` standing beside it |
| keeping the hot path inlined | `Q` staying in the table while `a, b` is frequent, and the pattern one level up that names the call with its variables |
