# An alternating pair, generalized

A worked case for [algorithm.md](algorithm.md): how `x, y, x, y`, `a, b, a, b` and `m, n, m, n` become one
function held once, by a neuron that keeps seeing a different pair in the same places. Nothing here is
normative. It follows one neuron's table through the greedy pick, the collapse of its residual, the price and
the election, and then the level above, where the function is held.

---

# 1. The stream

One event dimension, letters, laid out over time. The stream runs `x, y, x, y, p`, or `a, b, a, b, p`, or
`m, n, m, n, p`: an alternating pair, and then `p`. Three pairs recur; many others go by once. The pairs never
occur together. What they share is `p`, which follows every one of them, and `p` is the neuron this case
follows. What the letters build in their own tables is left out.

The levels are stated, not drawn: `p` is taken to stand at a level whose reach spans four frames (D4), and
offsets are written as plain frame counts.

# 2. The greedy pick: a constant per recurring pair

Every activation of `p` recorded the four letters before it (D7). The greedy pick (D33) builds a constant out of
whatever recurs, so each pair that came round twice gets one:

| in `p`'s table | the constant | its child |
|---|---|---|
| for `x, y` | `(p, 0)`, `(y, 1 back)`, `(x, 2 back)`, `(y, 3 back)`, `(x, 4 back)` | `XY` |
| for `a, b` | `(p, 0)`, `(b, 1 back)`, `(a, 2 back)`, `(b, 3 back)`, `(a, 4 back)` | `AB` |
| for `m, n` | `(p, 0)`, `(n, 1 back)`, `(m, 2 back)`, `(n, 3 back)`, `(m, 4 back)` | `MN` |

Each writes its occurrence as `p` and which of `p`'s patterns, where `p` and the letters cost five neurons. The
pairs seen once stay in the residual, written flat.

# 3. Parameters from what recurs without a name

Beside the three recurring pairs, many pairs have gone by once: `s, t, s, t, p`, `u, v, u, v, p`, and so on.
No letter recurs in them, so no pair of neurons reaches a count of two in the residual. What recurs is a
relation between offsets: one back and three back hold one neuron in every such row, and so do two back and
four back. Those are two parameters, each a pattern of its own, found by its own collapse and paying on its own
uses (D27, D33):

```
P:  (1 back, 3 back)        Q:  (2 back, 4 back)
```

`P` says that one back and three back hold the same letter, and `Q` the same for two and four back (D38). Each
writes two letters as one value, and neither names a letter or keeps a list of them.

# 4. The price

Say the machine holds thirty-two neurons, so a neuron costs 5 bits; `P` and `Q` have each passed eight letters,
so a value costs 3; and `p` holds five patterns, `XY`, `AB`, `MN`, `P` and `Q`, so which of them costs `log₂ 5`,
about 2.3 (D13).

| written as | bits |
|---|---|
| a once-seen pair, flat: `p` and four letters | 5 × 5 = 25 |
| by `P` and `Q`: `p` as itself, and each parameter as which pattern and which value | 5 + (2.3 + 3) + (2.3 + 3) = 15.6 |
| a recurring pair, by its constant: `p` and which pattern | 5 + 2.3 = 7.3 |

The parameters save 9.4 bits on every once-seen pair. Each one's line, an offset per cell and a neuron per value,
`2 · 2 + 8 · 5 = 44` bits, is paid once, and a letter new to `P` pays its entry the first time it passes. `XY`,
`AB` and `MN` stand while their pairs are frequent; when one becomes rare its constant retires (R18) and `P` and
`Q` cover that pair too, with nothing to learn first.

# 5. The return and the election

`p` bids `P` and `Q` whenever they are in its cover, each carrying the letter it held (D31). Nothing has a neuron
yet. The first time a bid is accepted the machine gives the parameter a child and a value neuron for the letter it
passed, `P`'s child and `P:t`, `Q`'s child and `Q:s`, one level above `p`, holding nothing (R16, R44).

# 6. One occurrence

`s, t, s, t, p` arrives, a pair never seen, and `p` fires.

| Stage | what happens |
|---|---|
| `process functions`, in `p` | No constant fits. `P` holds `t` at one and three back, and `Q` holds `s` at two and four. `p` bids `P` with `t` and `Q` with `s`. |
| the election | Each bid covers two letters, 10 bits, for 2.3 + 3, and is accepted. `p` itself stands as it is. |
| activate children | One level up, at `p`'s coordinate: `P`'s child and `P:t`, `Q`'s child and `Q:s` (§7.4). `p` stands beside them, uncovered. |

The level above sees `p`, the two children and the two value neurons together, siblings at offset zero (D26).
A constant there over `p` and the two children is the alternation, one neuron for every pair: `alternation`, a
level higher still. A constant over `p`, `P:t` and `Q:s` is the alternation of `t` and `s` in particular. The
first is built where the pairs vary; the second where one pair is frequent, which is what `XY` did a level down.

`s, t, u, t, p` does not fit fully: `P` holds `t` at both its offsets, but `Q` holds `s` at two back and `u` at
four back. `Q`'s value is the nearer, `s`, and four back is a failed cell (D44, D22).

# 7. What the children buy

`alternation` is one neuron whatever the pair was, so its connections are one set (D25): what the machine learns
to do after an alternation it has learned for every pair. `P:t` has a history and connections of its own, of
everything that followed an alternation whose first letter was `t`, so the level above can learn about a
particular pair where that matters, by a constant over `alternation` and `P:t`.

# 8. The same thing as code

| in the code | in the machine |
|---|---|
| three copies of one block with different names in them | `XY`, `AB` and `MN` in `p`'s table |
| noticing they are one block with two names that vary | the collapse finding four offsets that agree in pairs |
| the parameters | `P` and `Q`, patterns in `p`'s table |
| the extracted function | `alternation`, a constant one level up over `p` and the parameters' children |
| a call with its arguments | `alternation` firing, with `P:t` and `Q:s` below it |
| keeping the hot path inlined | `AB` staying in `p`'s table while `a, b` is frequent |
