# An alternating pair, generalized

A worked case for [algorithm.md](algorithm.md): how `x, y, x, y`, `a, b, a, b` and `m, n, m, n` become one
function held once, by a neuron that keeps seeing a different pair in the same places. Nothing here is
normative. It follows one neuron's history through the collapse, the price, the return and the election.

---

# 1. The stream

One event dimension, letters, laid out over time. The stream runs `x, y, x, y, p`, or `a, b, a, b, p`, or
`m, n, m, n, p`: an alternating pair, and then `p`. The pairs never occur together. What they share is `p`,
which follows every one of them, and `p` is the neuron this case follows. What the letters of the pairs build
in their own tables is left out.

The levels are stated, not drawn: `p` is taken to stand at a level whose reach spans four frames (D4), and
offsets are written as plain frame counts.

# 2. What `p` has seen

Every activation of `p` recorded one of three neighborhoods (D7):

| after | 1 back | 2 back | 3 back | 4 back |
|---|---|---|---|---|
| `x, y, x, y` | `y` | `x` | `y` | `x` |
| `a, b, a, b` | `b` | `a` | `b` | `a` |
| `m, n, m, n` | `n` | `m` | `n` | `m` |

Every place is filled every time, and no letter holds the majority at any of them.

# 3. The collapse writes slots

The greedy pick seeds on the place filled in the most residuals (D33), here any of the four, and the collapse
runs over all of `p`'s neighborhoods (D27).

| Question, per place | Answer |
|---|---|
| Does one neuron hold the majority there? | No, at all four. Nothing is named. |
| Is the place filled in a majority? | Yes, at all four. Each is a slot. |
| Which slots hold the same neuron in a majority of neighborhoods? | 1 back and 3 back; 2 back and 4 back. Two roles. |

The candidate:

```
(K¹, 1 back),  (J¹, 2 back),  (K¹, 3 back),  (J¹, 4 back)
```

`K¹` is whatever stood one back, and the same thing three back; `J¹` is whatever stood two back, and the same
thing four back (D38). The line says nothing about which letters they are.

# 4. The price

| | symbols |
|---|---|
| what it covers in one activation: `p` and four letters | 5 |
| what it costs there: its own line, and two bindings (D13) | 3 |
| saving per activation (D22) | 2 |
| its dictionary line, `1 + 4` | 5 |

Its margin is positive from the third activation on (D30), whichever pairs those three were. Three lines that
named the letters would each save four per activation, and each only over its own pair; a pair seen once could
have no line at all (R14).

# 5. The return

`p` adds the pattern to its table and joins it to the covers in its history where it pays, owning the letters
at the slot places there (R15). On the same return it asks the machine for the pattern's child, `alternation`,
and for a class neuron for each role, `K` and `J` (R41). The machine creates all three, one level above `p`,
holding nothing.

# 6. One occurrence

`a, b, a, b` has run, and `p` fires.

| Stage | what happens |
|---|---|
| `process functions`, in `p` | The line fits: `b` stands one back and three back, `a` two back and four back. `p` bids `alternation`, carrying `K¹ = b`, `J¹ = a` (D31). |
| the election | The bid covers `p` and the four letters for a price of three, and is accepted. |
| activate children | One level up, `alternation` fires at `p`'s coordinate with both bindings. `K` fires at the coordinate of the nearest `b`, bound to `b`, and `J` at the nearest `a`, bound to `a` (§7.4). |

The level above reads `alternation(b, a)`: a call and its two arguments.

# 7. A pair never seen

`s, t, s, t, p` arrives for the first time. `p`'s line fits at once: `t` stands one back and three back, `s` two
back and four back, and a slot takes whatever stands at its place. `alternation` fires with `K¹ = t`, `J¹ = s`.
Nothing about `s` or `t` had to be learned first.

`s, t, u, t, p` does not fit: `K¹` is the same neuron at both its offsets, but `J¹` would have to be `s` and
`u` at once. Both of `J¹`'s places count as named and absent, and the bid is priced accordingly (D22).

# 8. A pair that becomes frequent

If `x, y` comes to fill most of `p`'s history, `y` holds the majority one back and `x` two back, and the next
re-centering names them (D29): the line turns into the specific one, and the roles and their class neurons go
(R41). The pairs that are left fall to the residual, where the next greedy pick writes them as slots again.
Frequent pairs get a line of their own, and the rest go through the slots.

# 9. What the one child buys

`alternation` is one neuron whatever the pair was, so its connections are one set (D25): what the machine learns
to do after an alternation it has learned for every pair, with the bindings saying which pair this is. `K` and
`J` are neurons too. Each has a history pooling the neighborhoods around every letter it was bound to, and
connections of its own, and uncovered it votes (D41).

# 10. The same thing as code

| in the code | in the machine |
|---|---|
| one block that keeps recurring with different names in it | the neighborhoods in `p`'s history |
| noticing what changes and what does not | the collapse: named, a slot, or left out |
| the parameters | the roles `K¹` and `J¹` |
| the extracted function | `p`'s line, held once |
| a call with its arguments | `alternation` firing with `K¹ = b`, `J¹ = a`, and `K` and `J` beside it |
| specializing the hot path | the line re-centering onto `x, y` when that pair dominates |
