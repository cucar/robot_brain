# An alternating pair, generalized

A worked case for [algorithm.md](algorithm.md): how `x, y, x, y`, `a, b, a, b` and `m, n, m, n` become one
function held once, by a neuron that keeps seeing a different pair in the same places. Nothing here is
normative. It follows one neuron's table through the greedy pick, the merge, the price and the election.

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

# 3. The merge: three patterns become one with two variables

The merge (D42) pairs the two patterns of the table that differ at the fewest offsets. Any two of these differ
at all four, and no pair does better, so it takes the older two.

| Question | Answer |
|---|---|
| Where do they differ? | At all four offsets. |
| Which differing offsets hold one neuron in each pattern? | 1 back and 3 back (`y` in one, `b` in the other); 2 back and 4 back (`x` and `a`). |
| So how many class neurons? | Two: `K` at 1 and 3 back, `J` at 2 and 4 back. |

The candidate:

```
(K, 1 back),  (J, 2 back),  (K, 3 back),  (J, 4 back)
```

`K` is whatever stood one back, and the same thing three back; `J` is whatever stood two back, and the same
thing four back (D38). The pattern names no letter.

# 4. The price

The candidate is priced over the whole history (D30), not only where the pair it came from fits.

| written as | symbols |
|---|---|
| a pair, flat | 4 |
| a pair its own pattern covers: `Q` | 1 |
| a pair the candidate covers: `alternation`, `K`, `J` | 3 |

On the recurring pairs it can only lose: three symbols where `P`, `Q` and `R` write one. On the pairs seen once,
which nothing covered, it saves one each. Its dictionary line is five, and each class neuron one more (D13). So
the candidate pays once seven or so once-seen pairs have gone by, and the pick takes it. `P`, `Q` and `R` keep
their own margins: they stay while their pairs are frequent, and when one becomes rare its pattern retires (R18)
and the general pattern writes it.

# 5. The return and the election

`p` bids the general pattern whenever it is in a cover, with a class bid for `K` and one for `J` naming the
letters they were fit by (D31). Nothing has a neuron yet. The first time the bid is accepted the machine gives
the pattern a child, `alternation`, and each class a class neuron; no other bid of that election covers the same
ground or reads the same letters, so it creates all three, one level above `p`, holding nothing (R16, R41, R43).

# 6. One occurrence

`s, t, s, t, p` arrives, a pair never seen, and `p` fires.

| Stage | what happens |
|---|---|
| `process functions`, in `p` | No specific pattern fits. The general one does: `t` stands one back and three back, `s` two back and four back. `p` bids `alternation`, with class bids for `K` at the nearest `t` and `J` at the nearest `s`. |
| the election | The bid covers `p` and the four letters for a price of three, and is accepted. |
| activate children | One level up, `alternation` fires at `p`'s coordinate; a `K` variable fires at the `t`, holding `t`; a `J` variable fires at the `s`, holding `s` (§7.4). |

The level above reads `alternation` beside two variables, `K = t` and `J = s`. Nothing about `s` or `t` had to
be learned first.

`s, t, u, t, p` does not fit: `K` is the same value at both its offsets, but `J` would have to hold `s` and `u`
at once. Both of `J`'s offsets count as named and absent, and the bid is priced accordingly (D22).

# 7. What the one child buys

`alternation` is one neuron whatever the pair was, so its connections are one set (D25): what the machine learns
to do after an alternation it has learned for every pair, and the variables standing beside it say which pair
this was. `K` and `J` are neurons too. Each has a history pooling the neighborhoods around every letter it has
held, and connections of its own, and uncovered it votes (D41).

# 8. The same thing as code

| in the code | in the machine |
|---|---|
| three copies of one block with different names in them | `P`, `Q` and `R` in `p`'s table |
| noticing they are one block with two names that vary | the merge pairing two of them |
| the variables | the class neurons `K` and `J` |
| the extracted function | the general pattern, held once in `p`'s table |
| a call, with the variables it reads | `alternation` firing, with `K` and `J` standing beside it |
| keeping the hot path inlined | `Q` staying in the table while `a, b` is frequent |
