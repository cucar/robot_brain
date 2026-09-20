# An alternating pair, generalized

A worked case for [algorithm.md](algorithm.md): how `x, y, x, y`, `a, b, a, b` and `m, n, m, n` become one
function held once. Nothing here is normative. It follows the two forces of a level in turn: separating, which
gives each neuron a function of its own, and grouping, which notices that the functions are one function under
different names and makes the class neurons that let it be written once.

---

# 1. The stream

One event dimension, letters, laid out over time. At different times the stream runs `x, y, x, y, x, y, …`, or
`a, b, a, b, …`, or `m, n, m, n, …`. Nothing else connects the three pairs: they never occur together, and
nothing in the stream follows all of them.

The levels are stated, not drawn: the neurons below are taken to stand at a level whose reach spans three
frames (D4), and offsets are written as plain frame counts.

# 2. Separating: each neuron learns its own function

This is ordinary `process functions` (§6). Every time `y` fires it sees `x` one back, `y` two back and `x` three
back, so the collapse over `y`'s history names all three (D27), the line pays for itself after two occurrences
(D30), and `y` has a pattern with a child. `x` learns the mirror of it, and so do the other four.

| in the table of | the line | its child |
|---|---|---|
| `y` | `(x, 1 back)`, `(y, 2 back)`, `(x, 3 back)` | `P` |
| `x` | `(y, 1 back)`, `(x, 2 back)`, `(y, 3 back)` | `P′` |
| `b` | `(a, 1 back)`, `(b, 2 back)`, `(a, 3 back)` | `Q` |
| `a` | `(b, 1 back)`, `(a, 2 back)`, `(b, 3 back)` | `Q′` |
| `n` | `(m, 1 back)`, `(n, 2 back)`, `(m, 3 back)` | `R` |
| `m` | `(n, 1 back)`, `(m, 2 back)`, `(n, 3 back)` | `R′` |

No one of these histories contains any variation. In `y`'s, the neighbor one back is always `x`. That is why
no neuron can see, from its own evidence, that the six of them are alike: the variation is between the lines,
not inside any of them.

# 3. Grouping: the machine notices the lines are one line

In its next `process classes` call each neuron reports the line it added (§6.6). The machine computes each
line's shape (R42): the names out, the roles numbered by first appearance, the owner role `0`.

| line | shape |
|---|---|
| `y`'s: `x`, `y`, `x` | role 1, role 0, role 1 |
| `b`'s: `a`, `b`, `a` | role 1, role 0, role 1 |
| `x`'s: `y`, `x`, `y` | role 1, role 0, role 1 |
| … | the same |

All six land in one bucket. The machine lines them up position by position. No position has the same name in
every line, so every position is a role. Role 0 collects the owners, `y, x, b, a, n, m`. Role 1 collects what
stood one back, `x, y, a, b, m, n`, which is the same six. Same members, same class neuron: the machine mints
one, `K`, and tells each of the six it has joined.

So the kind is "a letter that alternates with a partner", and the general line needs one class and two roles:

```
(K¹, 1 back),  (K⁰, 2 back),  (K¹, 3 back)
```

`K⁰` is the owner, whoever it is; `K¹` is its partner, whoever that is; and `K¹` named twice is the same partner
both times (D38).

# 4. The class learns the function itself

The machine wrote no pattern. It minted `K` and delivered membership, and the rest is `K` being a neuron.

From its next activation on, each of the six reports `K`, so `K` fires beside `y`, beside `b`, beside `n`, and
gets its own `process functions` call each time with the neighborhood around that member. One back in that
pooled history stand `x`, `a`, `m` by name, none with a majority, and `K` every time. Two back stands `K` again,
with the owner's own binding. So `K`'s ordinary greedy pick and collapse (D33, D27) produce the line of §3, and
`K`'s own margin prices it (D30). Its child is `alternation`.

# 5. One occurrence, through the class

`a, b, a, b` is running, and `b` fires.

| Stage | what happens |
|---|---|
| `process classes` | `b` refreshes its history and reports `K`. So did `a`, a frame ago, and `b`, two frames ago. |
| bind classes | The machine adds a `K` activation beside `b`, bound to `b`. The ones beside the earlier `a` and `b` are already standing. |
| `process functions`, in `b` | `b`'s own line fits. It bids `Q`. |
| `process functions`, in `K` | `K`'s line fits: `K¹` one back is bound to `a`, `K⁰` two back to `b`, `K¹` three back to `a` again, the same partner. It bids `alternation`, carrying `K⁰ = b`, `K¹ = a`. |
| the election | Both bids cover the same four activations. |

# 6. Which one is written

| written as | symbols |
|---|---|
| flat: `a`, `b`, `a`, `b` | 4 |
| `b`'s child `Q` | 1 |
| `alternation` with `K⁰ = b`, `K¹ = a` | 3 |

`Q` covers the same ground for a price of one against three, so while `Q` lives it wins the election and the
occurrence is written as `Q`. The general line is priced on `K`'s own books, against the flat run, where it
saves one per occurrence and pays for its four symbols after five; it never learns that `Q` took the credit
(R24). So the dictionary holds both.

When `a, b` becomes rare, `Q`'s margin goes negative on `b`'s books and `Q` retires (R18). From then on the same
occurrence is written as `alternation`, three symbols instead of four. Frequent pairs keep a symbol of their
own; the rest go through the rule.

# 7. What the one child buys

`alternation` is one neuron whatever the pair was, so its connections are one set (D25). What the machine
learns to do after an alternation it has learned for every pair that is a member, with the bindings saying
which pair this is. `P`, `Q` and `R` are three neurons with three sets of lessons and cannot share any of them.

# 8. The same thing as code

| in the code | in the machine |
|---|---|
| three copies of one block with different names in them | the lines in `y`'s, `b`'s and `n`'s tables |
| noticing they are the same block | the machine's bucket receiving its second line of one shape |
| the parameters | the roles `K⁰` and `K¹` |
| the extracted function | `K`'s line, held once, in `K`'s table |
| a call with its arguments | `alternation` firing with `K⁰ = b`, `K¹ = a` |
| keeping the hot path inlined | `Q` winning the election while `a, b` is frequent |
