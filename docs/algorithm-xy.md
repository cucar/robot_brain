# An alternating pair, generalized

A worked case for [algorithm.md](algorithm.md): how `x, y, x, y`, `a, b, a, b` and `m, n, m, n` become one
function held once, by a neuron that keeps seeing a different pair in the same places. Nothing here is
normative. It follows one neuron's table through the greedy pick, the classes found in its residual, the price
and the election.

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

# 3. Classes from the residual

Beside the three recurring pairs, many pairs have gone by once: `s, t, s, t, p`, `u, v, u, v, p`, and so on.
Nothing recurs by name, so the greedy pick built nothing for them and they stand in the residual, flat. Before
`p`'s next pick seeds on a neighbor of theirs, clusters the rows it collects, and surveys the cluster of once-seen
pairs spot by spot (D33, D42):

| Question | Answer |
|---|---|
| Which spots hold one neuron in a majority? | None. No constants. |
| Which spots are filled in a majority with no neuron in a majority? | All four. Each is a slot. |
| Which slots hold the same neuron in a majority of the neighborhoods? | One and three back; two and four back. So `K¹` at both of the first, `J¹` at both of the second: two classes, two variables. |

The candidate is the pattern and its two classes at once, `K` holding `t, v, …` and `J` holding `s, u, …`:

```
(K¹, 1 back),  (J¹, 2 back),  (K¹, 3 back),  (J¹, 4 back)
```

`K¹` is whatever member of `K` stood one back, and the same one three back; `J¹` likewise (D38). The pattern
names no letter.

# 4. The price

| written as | symbols |
|---|---|
| a once-seen pair, flat | 4 |
| a recurring pair, by its own pattern: `Q` | 1 |
| a once-seen pair, by the general pattern: `alternation`, `K¹`, `J¹` | 3 |

The general pattern saves one on every once-seen pair and nothing on the recurring ones, which it does not
cover; its line costs five and each class its name and its members (D13), so it pays once enough rare pairs
have gone by. `P`, `Q` and `R` stand: they differ from the general pattern at places where no class stands
beside their letters, since `y`, `b` and `n` were never in the residual. When x-y becomes rare and `P` retires,
`y` and `x` fall to the residual, a survey finds them at `K`'s and `J`'s slots, the classes hold them (D41), and
the general pattern writes the pair from then on.

# 5. The return and the election

`p` bids the general pattern whenever it is in a cover, with a class bid for `K` and one for `J`, each carrying
the class's members and naming the letter it was fit by (D31). Nothing has a neuron yet. The first time the bid is accepted the machine gives
the pattern a child, `alternation`, and each class a class neuron; no other bid of that election covers the same
ground or reads the same letters, so it creates all three, one level above `p`, holding nothing (R16, R41, R43).

# 6. One occurrence

`s, t, s, t, p` arrives for the second time, `s` and `t` now members of `J` and `K`, and `p` fires.

| Stage | what happens |
|---|---|
| `process functions`, in `p` | `K` stands beside each `t` and `J` beside each `s`. No specific pattern fits. The general one does: the same member of `K` one back and three back, the same member of `J` two and four back. `p` bids `alternation`, with class bids for `K` at the nearest `t` and `J` at the nearest `s`. |
| the election | The bid covers `p` and the four letters for a price of three, and is accepted. |
| activate children | One level up, `alternation` fires at `p`'s coordinate; a `K` variable fires at the `t`, holding `t`; a `J` variable fires at the `s`, holding `s` (§7.4). |

The level above reads `alternation` beside two variables, `K = t` and `J = s`. The first time the pair came by
it was written flat, and `p` found `s` and `t` standing where `J` and `K` stand and made them members; that is
all that had to be learned.

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
| noticing they are one block with two names that vary | the residual grouped by offset into `K` and `J` |
| the variables | the class neurons `K` and `J` |
| the extracted function | the general pattern, held once in `p`'s table |
| a call, with the variables it reads | `alternation` firing, with `K` and `J` standing beside it |
| keeping the hot path inlined | `Q` staying in the table while `a, b` is frequent, and the pattern one level up that names the call with its variables |
