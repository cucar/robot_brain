# Addition, as a machine that could be built

A worked case for [algorithm.md](algorithm.md): a machine that adds two binary numbers, written out neuron by
neuron. Nothing here is normative, and the machine is shown running: its tables and connections are as listed,
and its histories already hold sums like this one. It shows the parts of the design doing one small job: a
request with each column; a neuron for each case that can be asked; a step per case that writes the answer; and
a carry that the world shows while it teaches and the machine expects afterwards.

---

# 1. The environment

**In.** Each frame the world presents one column of the sum, the two numbers read from the right: the event
`add`, which is the request, the digit of the first number in event dimension `a`, and the digit of the second
in event dimension `b`. Each digit dimension has two buckets, `0` and `1`. When the numbers run out the world
fires `add` once more with no digits, asking for the last digit.

**Out.** One action dimension, `out`, with two base actions, `out-0` and `out-1`. What runs there, in the frame
after a request, is the next digit of the result, from the right.

**Teaching.** While the machine is being taught, the world runs the right digit itself in the frame after each
request, and rewards it (§3.5). Where the column carried, the world also shows the event `carry` in that frame,
beside the next request. It teaches one case at a time: the same column, with or without a carry coming in, over
and over, before it moves to the next. Afterwards the world runs nothing and shows no carry.

# 2. The carry

`carry` is an event. In training the world shows it. After training nothing reports it: the step that answers a
carrying column is expanded, and the expansion places `carry` in the frame the answer runs in, weakly, as an
expected event (R30). The world reports nothing in its place, so the expectation is what the machine has in
that frame, and a weak activation is an input of its frame like a strong one (D40). That frame is also the
frame of the next request, so `add` sees `carry` beside the next column's digits. Writing the carry is expecting
it; reading it is seeing it beside the next request.

# 3. The neurons

| Kind | Neurons |
|---|---|
| base events | `add`, `a0`, `a1`, `b0`, `b1`, `carry` |
| base actions | `out-0`, `out-1` |
| the cases | nine functions in `add`'s table, each naming what stands beside the request: the two digits, and `carry` where one came in; each with a child |
| the steps | nine functions, in the tables of `out-0` and `out-1`, each naming its case's child a frame before, and `carry` beside it where the case carries; each with a child |

# 4. How it was learned

1. **A request and its column become one neuron.** While one case is taught, the same neurons stand beside `add`
   every time, so `add`'s table holds a function that names them outright (D47, D27), and its child is that
   case: `1 + 1` asked, or `0 + 1` asked with a carry. Nothing varied while it was learned, so no variable
   formed, and what a function names is not open to a variable afterwards (D19). Eight columns give eight
   cases, and the request with no digits and a carry is the ninth.
2. **A carry tells two cases apart.** `1 + 1` with a carry and `1 + 1` without are two functions. Where `carry`
   stands, the one that names it covers more and is taken; where it does not, the one that names it would be
   written with a failed neighbor, and the other is taken (D28).
3. **The case learns its answer.** In the taught frames the right digit ran one frame after the case and was
   rewarded. The case's child was open and uncovered, so it connected to that digit, with the reward (R31).
4. **The answer gets its step.** In the digit's history the case's child stands one back in every such row, and
   `carry` beside it where the column carried, so its table holds a function naming them, the step. From then
   on the case is covered in the frame the answer runs, and in that frame it connects to the step, the neuron
   that covered it, with that frame's reward (D10).

| Case, beside `add` | its step, in the table of | names beside it |
|---|---|---|
| `0`, `0` | `out-0` | |
| `0`, `1` | `out-1` | |
| `1`, `0` | `out-1` | |
| `1`, `1` | `out-0` | `carry` |
| `0`, `0`, `carry` | `out-1` | |
| `0`, `1`, `carry` | `out-0` | `carry` |
| `1`, `0`, `carry` | `out-0` | `carry` |
| `1`, `1`, `carry` | `out-1` | `carry` |
| `carry` alone | `out-1` | |

Inferred, a step is expanded (R28): its bidder, the digit, is output, and `carry` is placed beside it where the
step names it, as what the machine expects to see.

A last request with no digits and no carry has nothing beside it to name. `add` then stands uncovered and
speaks from its own connections, most of them written in exactly those frames, and they say `out-0`.

# 5. One sum, frame by frame, after teaching

`101 + 111`. The columns, from the right, are `(1, 1)`, `(0, 1)`, `(1, 1)`. What is chosen in one frame runs in
the next (R29).

| Frame | the world shows | runs, and is expected | the case on the apex | infers |
|---|---|---|---|---|
| 1 | `add`, `1`, `1` | | `1, 1` | the step `1, 1 → out-0, carry` |
| 2 | `add`, `0`, `1` | `out-0`; `carry` expected | `0, 1, carry` | the step `0, 1, carry → out-0, carry` |
| 3 | `add`, `1`, `1` | `out-0`; `carry` expected | `1, 1, carry` | the step `1, 1, carry → out-1, carry` |
| 4 | `add` | `out-1`; `carry` expected | `carry` alone | the step `carry → out-1` |
| 5 | | `out-1` | | |

The digits that ran in `out`, from the right: `0`, `0`, `1`, `1`. The result is `1100`, a sum the machine need
never have seen: every column is one of the nine cases, and the carry takes each column's outcome into the
next.

# 6. Taught in another order

Had the columns been mixed from the first lesson, the digits beside `add` would have varied before any case
recurred, and variables are built first (D33): `add`'s table would hold a variable for each digit and for the
carry, and one function over them. What fires then is one child for every column, with a value neuron for each
digit beside it. A neuron for the column as a whole still forms, above them: each value neuron's own table names
what always stands beside it, and a level at a time the values are put back together into one neuron
([algorithm-evaluation.md](algorithm-evaluation.md)). The machine then learns the same lessons on that neuron,
several levels up and later. Taught a case at a time, it has a neuron per case at the first level.

# 7. The same algorithm as code

Written as code, before this design existed, the algorithm was recursive rather than looping. Each of its
parts has a place here:

| in the code                           | in the machine                                                |
|---------------------------------------|---------------------------------------------------------------|
| calling the function                  | `add`, fired with each column                                 |
| `addDigits(d1, d2)`, a table of cases | the nine cases, and the step each has learned                 |
| `getDigit(num, pos)`                  | the world presenting the next column                          |
| the `carry` parameter                 | `carry`, shown while teaching and expected afterwards         |
| recursion over `pos`                  | the frame: each request is one of the same cases              |
| the base case: no digits, no carry    | the frame in which nothing is asked, so nothing runs          |
