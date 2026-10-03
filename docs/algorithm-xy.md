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

What stands under `p` is stated, not drawn: `p` is taken to be a neuron that sees four frames back (D4), and
offsets are written as plain frame counts.

# 2. Variables first: two parameters

Every activation of `p` recorded the four letters before it (D7). The greedy pick builds variables first (D33).
In every row, whatever the pair, one back and three back hold one neuron, and so do two back and four back. A
pair of offsets that agree is a relation a variable is seeded on (D47), and each is grown by its own collapse
and kept on its own uses (D27). A variable over several positions is called a parameter (D41):

```
P:  (1 back, 3 back)        Q:  (2 back, 4 back)
```

`P` says that one back and three back hold the same letter, and `Q` the same for two and four back (D38). Each
writes two letters as one member, and its members are the letters that have stood at both its offsets; neither
keeps a list of the pairs. They form as soon as
alternating pairs have come round twice, the same pair or two different ones.

# 3. The function over them

Then the pick builds functions, over the rows as the variables have rewritten them. `P` and `Q` hold together in
every row, which is a relation between two variables, and the collapse grows it into a function (D27):

| Question | Answer |
|---|---|
| Does any letter pay as a fixed cell? | No. Over all the rows, no letter stands at any offset often enough. |
| Which variables hold in those rows? | `P`, over one and three back; `Q`, over two and four back. |
| So what does the function name? | The two parameters, and nothing else. |

```
alternation:  P, Q
```

The function fixes no letter. It says that `P` and `Q` held together, which is the alternation, whatever the
pair was.

# 4. A function per recurring pair

Three pairs recur. In the rows of `x, y` the same letters stand at all four offsets every time, and there a
fixed cell saves more on each than a parameter does, since a parameter pays for which member on every occurrence.
So on those rows the pick also builds a function with four fixed cells, which takes the cells from `P` and `Q`
there (D27):

| in `p`'s table | fixed cells | its child |
|---|---|---|
| for `x, y` | `(y, 1 back)`, `(x, 2 back)`, `(y, 3 back)`, `(x, 4 back)` | `XY` |
| for `a, b` | `(b, 1 back)`, `(a, 2 back)`, `(b, 3 back)`, `(a, 4 back)` | `AB` |
| for `m, n` | `(n, 1 back)`, `(m, 2 back)`, `(n, 3 back)`, `(m, 4 back)` | `MN` |

Each is taken once its pair has recurred enough that the two members it stops writing pay for its line (D30).
The pairs seen once are covered by `alternation`.

# 5. The price

Say the machine holds thirty-two neurons, so a neuron costs 5 bits; `P` and `Q` each hold eight letters as members,
so which member costs 3; and `p` holds six patterns, `P`, `Q`, `alternation`, `XY`, `AB` and `MN`, so which of them
costs `log₂ 6`, about 2.6 (D13).

| written as | bits |
|---|---|
| a once-seen pair, flat: `p` and four letters | 5 × 5 = 25 |
| by `P` alone, the other two letters flat: `p`, which pattern, one member, two letters | 5 + 2.6 + 3 + 5 + 5 = 20.6 |
| by `alternation`: `p`, which pattern, and two members | 5 + 2.6 + 3 + 3 = 13.6 |
| a recurring pair, by its own function: `p` and which pattern | 5 + 2.6 = 7.6 |

A second parameter bidding alone does not pay here: it would write `p` and which pattern again, 10.6 bits, for
two letters worth 10. So without the function a once-seen pair costs 20.6 bits, and `alternation` saves 7 on
every one (D30). Its line is two references, which pattern each, 5.2 bits, so it pays at its first occurrence.
Each parameter's line, an offset per position and a neuron per member, `2 · 2 + 8 · 5 = 44` bits, is paid once,
and a letter new to `P` joins the first time it passes, since standing at both offsets pays for its entry
(D27). A recurring pair's own function saves a further 6 bits on each occurrence, the two members, for a line
of four fixed cells, `4 · (5 + 2) = 28` bits, so it is built once its pair has come round five times within the
history. `XY`, `AB` and `MN` stand while their pairs are frequent; when one becomes rare its function retires
(R18) and `alternation` writes that pair too, with nothing to learn first.

# 6. The return and the election

`p` sends `alternation` whenever it is in a cover, as one bid with `P` and `Q` and the letter each held (D31).
Nothing has a neuron yet. The first time the bid is accepted the machine gives the function a child,
`alternation`, and each parameter a value neuron for the letter it passed, `P:t` and `Q:s`, one level above
`p`, holding nothing (R16, R44).

# 7. One occurrence

`s, t, s, t, p` arrives, a pair never seen, and `p` fires.

| Stage | what happens |
|---|---|
| `process functions`, in `p` | No function with fixed cells fits. `P` holds `t` at one and three back and `Q` holds `s` at two and four, so `alternation` holds. `p` sends one bid: the function, with `P = t` and `Q = s`. |
| the election | The bid covers `p` and the four letters, 25 bits, for a price of 13.6, and is accepted. |
| activate children | One level up, at `p`'s coordinate: `alternation`, `P:t` and `Q:s` (§7.4). |

The level above reads three neurons at one coordinate: which function ran, and what it was given. Nothing about
`s` or `t` had to be learned first: each joined its parameter in this row, and only their value neurons are new.

`s, t, u, t, p` does not fit fully: `P` holds `t` at both its offsets, but `Q` holds `s` at two back and `u` at
four back. `Q`'s value is the nearer, `s`, and four back is a failed cell (D41, D22).

# 8. What the one child buys

`alternation` is one neuron whatever the pair was, so its connections are one set (D25): what the machine learns
to do after an alternation it has learned for every pair, a pair never seen before included. The value neurons
standing beside it say which pair this was. Each has a history and connections of its own, `P:t` of everything
that followed an alternation whose first letter was `t`, so the level above can learn about a particular pair
where that matters, by a function over `alternation` and `P:t` together.

# 9. The same thing as code

| in the code | in the machine |
|---|---|
| three copies of one block with different names in them | `XY`, `AB` and `MN` in `p`'s table |
| noticing they are one block with two names that vary | four offsets that agree in pairs: the two parameter relations |
| the parameters | `P` and `Q`, in `p`'s table |
| the extracted function | `alternation`, a function that names `P` and `Q`, held once in `p`'s table |
| a call with its arguments | `alternation` firing, with `P:t` and `Q:s` beside it |
| keeping the hot path inlined | `XY` staying in the table while `x, y` is frequent |
