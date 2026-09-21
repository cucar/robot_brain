# Addition, as a machine that could be built

A worked case for [algorithm.md](algorithm.md): a machine that adds two binary numbers, written out neuron by
neuron. Nothing here is normative, and the machine is shown running: its tables and connections are as listed,
and its histories already hold sums like this one. It shows the parts of the design doing one small job: event patterns that
return actions, an action that returns an event, and that returned event serving as the machine's only
variable, the carry.

---

# 1. The environment

**In.** Each frame the environment presents one pair of digits, the two numbers read from the right: the digit of
the first number in event dimension `a`, the digit of the second in event dimension `b`. Each dimension has two
buckets, `0` and `1`. When the numbers run out it presents nothing.

**Out.** One action dimension, `out`, with two base actions, `out-0` and `out-1`. What runs there is the next
digit of the result, from the right.

The environment holds nothing else. There is no sheet, no position and no carry outside the machine.

# 2. The carry

The carry is internal. A second action dimension, `carry`, has one base action, `carry-1`. The environment
does not have that dimension, so `carry-1` runs and nothing outside the machine sees it (D37). It
returns the event `c` (D39): in the frame `carry-1` runs, `c` fires weakly, beside that frame's pair of digits,
there for that frame and gone. Writing the carry is running `carry-1`; reading it is recognizing `c`. That is
the whole of the variable.

# 3. The neurons

| Kind | Neurons |
|---|---|
| base events | `a0`, `a1`, `b0`, `b1`, and the returned `c` |
| base actions | `out-0`, `out-1`, `carry-1` |
| patterns | eight, in the tables of `a0` and `a1`, each with a child |

# 4. The situations, and what each returns

A situation is a pattern in the table of the `a` digit that fired, naming what fired beside it in the same
frame. Offsets are all zero: the pair and the carry arrive together.

| Pattern, in the table of | names | returns in `out` | returns in `carry` |
|---|---|---|---|
| `a0` | `(b0, 0)` | `out-0` | |
| `a0` | `(b1, 0)` | `out-1` | |
| `a1` | `(b0, 0)` | `out-1` | |
| `a1` | `(b1, 0)` | `out-0` | `carry-1` |
| `a0` | `(b0, 0)`, `(c, 0)` | `out-1` | |
| `a0` | `(b1, 0)`, `(c, 0)` | `out-0` | `carry-1` |
| `a1` | `(b0, 0)`, `(c, 0)` | `out-0` | `carry-1` |
| `a1` | `(b1, 0)`, `(c, 0)` | `out-1` | `carry-1` |

What a situation returns is its child's action connections (D25), one per action dimension, each at offset one.
When `c` is present, both the pattern that names it and the one that does not fit the frame; the cover takes
the one that names it, since it covers more for the same price (D28), and its child is the one on the apex.

**The last carry.** The base neuron `c` holds one connection of its own, `out-1`. In a frame with a pair of
digits a situation covers `c`, and a covered neuron says nothing (D10). In the frame after the numbers run out
nothing covers it, so it speaks, and the final `1` is written.

# 5. One sum, frame by frame

`101 + 111`. The pairs, from the right, are `(1, 1)`, `(0, 1)`, `(1, 1)`. What is chosen in one frame runs in the
next (R29), and a return fires in the frame its call runs.

| Frame | runs | input | situation on the apex | chooses |
|---|---|---|---|---|
| 1 | | `a1`, `b1` | `1 · 1` | `out-0`, `carry-1` |
| 2 | `out-0`, `carry-1` | `a0`, `b1`, `c` | `0 · 1 · c` | `out-0`, `carry-1` |
| 3 | `out-0`, `carry-1` | `a1`, `b1`, `c` | `1 · 1 · c` | `out-1`, `carry-1` |
| 4 | `out-1`, `carry-1` | `c` | `c`, uncovered | `out-1` |
| 5 | `out-1` | | | |

The digits that ran in `out`, from the right: `0`, `0`, `1`, `1`. The result is `1100`.

# 6. The same algorithm as code

Written as code, before this design existed, the algorithm was recursive rather than looping. Each of its
parts has a place here:

| in the code                           | in the machine                                                |
|---------------------------------------|---------------------------------------------------------------|
| `addDigits(d1, d2)`, a table of cases | the eight situations and what each returns                    |
| `getDigit(num, pos)`                  | the environment presenting the next pair                      |
| the `carry` parameter                 | the returned event `c`                                        |
| recursion over `pos`                  | the frame: each pair is one call of the same eight situations |
| the base case: no digits, no carry    | the frame in which nothing fires, so nothing runs             |
