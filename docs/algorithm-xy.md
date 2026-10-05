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
and kept on its own uses (D27). Positions that hold one and the same neuron are a parameter (D44):

```
P:  (1 back, 3 back)        Q:  (2 back, 4 back)
```

`P` says that one back and three back hold the same letter, and `Q` the same for two and four back (D38). Each
writes two letters as one member, and its members are the letters that have stood at both its offsets, a record of
what has passed and no condition on what may; neither keeps a list of the pairs. They form as soon as alternating
pairs have come round twice, the same pair or two different ones.

# 3. The function over them

Then the pick builds functions, over the rows as the variables have rewritten them. `P` and `Q` hold together in
every row, which is a relation between two variables, and the collapse grows it into a function (D27):

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

# 4. The recurring pairs

Three pairs recur, and they are written by the same function as every other pair. Once `P` and `Q` hold those
offsets, the letters there are theirs, and no function fixes a letter in their place (D27). What is particular
to a recurring pair is already on the level above: `x, y` fires `alternation` with `P:y` and `Q:x` beside it,
the same three neurons every time, and a function there over the three is that pair as one neuron.

# 5. The price

Say the base alphabet has thirty-two symbols, so a letter costs 5 bits to write, and say `p` cost 5 as well.
`P` and `Q` each hold eight letters as members, so which member costs 3; and `p` holds three patterns, `P`, `Q`
and `alternation`, so which of them costs `log₂ 3`, about 1.6 (D13).

| written as | bits |
|---|---|
| a pair, flat: `p` and four letters, each at what it cost | 5 × 5 = 25 |
| by `P` alone, the other two letters flat: `p`, which pattern, one member, two letters | 5 + 1.6 + 3 + 5 + 5 = 19.6 |
| by `alternation`: `p`, which pattern, and two members | 5 + 1.6 + 3 + 3 = 12.6 |

A second parameter bidding alone would write `p` and which pattern again, 9.6 bits, for two letters that cost
10, so it barely pays. Without the function a pair costs about 19.6 bits, and `alternation` saves 7 on every one
(D30). Its line is two references, which pattern each, 3.2 bits, so it pays at its first occurrence. A
parameter's line is an offset per position and a neuron per member, and naming a neuron in a line is a choice
among every neuron the machine holds: with two hundred and fifty-six of them, `2 · 2 + 8 · 8 = 68` bits, paid
once. A letter new to `P` is its value the first time it passes, since agreeing at both offsets is the whole
test (D27), and is a member from then on.

What the call fires stands on the frontier at what it cost to write: `alternation` at 6.6 bits, the owner and
which pattern, and `P:t` and `Q:s` at 3 bits each. Whatever covers them later saves that and no more (D13).

# 6. The return and the election

`p` sends `alternation` whenever it is in a cover, as one bid with `P` and `Q` and the letter each held (D31).
Nothing has a neuron yet. The first time the bid is accepted the machine gives the function a child,
`alternation`, and each parameter a value neuron for the letter it passed, `P:t` and `Q:s`, one level above
`p`, holding nothing (R16, R44).

# 7. One occurrence

`s, t, s, t, p` arrives, a pair never seen, and `p` fires.

| Stage | what happens |
|---|---|
| `process functions`, in `p` | `P` holds `t` at one and three back and `Q` holds `s` at two and four, so `alternation` holds. `p` sends one bid: the function, with `P = t` and `Q = s`. |
| the election | The bid covers `p` and the four letters, 25 bits, for a price of 12.6, and is accepted. |
| activate children | One level up, at `p`'s coordinate: `alternation`, `P:t` and `Q:s` (§7.4). |

The level above reads three neurons at one coordinate: which function ran, and what it was given. Nothing about
`s` or `t` had to be learned first: each joined its parameter in this row, and only their value neurons are new.

`s, t, u, t, p` does not fit fully: `P` holds `t` at both its offsets, but `Q` holds `s` at two back and `u` at
four back. `Q`'s value is the nearer, `s`, and four back is a failed neighbor (D44, D22).

# 8. What the one child buys

`alternation` is one neuron whatever the pair was, so its connections are one set (D25): what the machine learns
to do after an alternation it has learned for every pair, a pair never seen before included. The value neurons
standing beside it say which pair this was. Each has a history and connections of its own, `P:t` of everything
that followed an alternation whose first letter was `t`, so the level above can learn about a particular pair
where that matters, by a function over `alternation` and `P:t` together.

# 9. The same thing as code

| in the code | in the machine |
|---|---|
| three copies of one block with different names in them | `x, y, x, y`, `a, b, a, b` and `m, n, m, n` in the stream |
| noticing they are one block with two names that vary | four offsets that agree in pairs: the two parameter relations |
| the parameters | `P` and `Q`, in `p`'s table |
| the extracted function | `alternation`, a function that names `P` and `Q`, held once in `p`'s table |
| a call with its arguments | `alternation` firing, with `P:t` and `Q:s` beside it |
| keeping the hot path inlined | a function one level up over `alternation`, `P:y` and `Q:x`, while `x, y` is frequent |
