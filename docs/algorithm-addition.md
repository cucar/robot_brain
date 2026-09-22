# Addition, as a machine that could be built

A worked case for [algorithm.md](algorithm.md): a machine that adds two binary numbers, written out neuron by
neuron. Nothing here is normative, and the machine is shown running: its tables and connections are as listed,
and its histories already hold sums like this one. It shows the parts of the design doing one small job: a step
per case, naming the digits it saw and the actions it ran; a connection from the digits to the step; and the
step's expected outcome, the carry, serving as the machine's only variable.

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
does not have that dimension, so `carry-1` runs and nothing outside the machine sees it (D37). While the sums
were being taught, a carry mark `c` was shown in the frame after every `carry-1`, so the steps that ran
`carry-1` connected to `c` a frame on (D25). Now the mark is no longer shown and the expectation stands in for
it (R40): in the frame after `carry-1` runs, `c` fires weakly, beside that frame's pair of digits, there for
that frame and gone. Writing the carry is running `carry-1`; reading it is recognizing `c`. That is the whole
of the variable.

# 3. The neurons

| Kind | Neurons |
|---|---|
| base events | `a0`, `a1`, `b0`, `b1`, and the returned `c` |
| base actions | `out-0`, `out-1`, `carry-1` |
| patterns | eight steps, in the tables of `out-0` and `out-1`, each naming the digits of the frame before and the actions of its own, each with a child |

# 4. The situations, and what each returns

A step is a pattern in the table of the `out` digit that ran, naming the pair of digits, and the carry if there
was one, a frame before it, and `carry-1` beside it when it ran (D5). Matched, it says this case happened and
this was written for it; inferred, it writes the digit and, where it names `carry-1`, carries (R30).

| Step, in the table of | names a frame before | names beside it | expects a frame on |
|---|---|---|---|
| `out-0` | `a0`, `b0` | | |
| `out-1` | `a0`, `b1` | | |
| `out-1` | `a1`, `b0` | | |
| `out-0` | `a1`, `b1` | `carry-1` | `c` |
| `out-1` | `a0`, `b0`, `c` | | |
| `out-0` | `a0`, `b1`, `c` | `carry-1` | `c` |
| `out-0` | `a1`, `b0`, `c` | `carry-1` | `c` |
| `out-1` | `a1`, `b1`, `c` | `carry-1` | `c` |

What a frame of digits infers is its connections (D25): the digits that stood uncovered on the apex connected,
a frame on, to the step that followed them, and that step's estimate is what the sums earned. The step is
expanded (R28): `out-0` or `out-1` runs, `carry-1` runs where the step names it, and `c` fires weakly the frame
after where the step's own connection expects it (R40). When `c` is present, both the step that names it and
the one that does not can be inferred; the one that names it has the higher estimate, since it is the one that
ran on the sums where `c` stood.

**The last carry.** The base neuron `c` holds one connection of its own, to the step `out-1` alone, learned on
the frame after every sum whose last column carried. In a frame with a pair of digits, the pair and `c` together
infer a step of the table above; in the frame after the numbers run out only `c` stands, so only its connection
speaks, and the final `1` is written.

# 5. One sum, frame by frame

`101 + 111`. The pairs, from the right, are `(1, 1)`, `(0, 1)`, `(1, 1)`. What is chosen in one frame runs in the
next (R29), and a return fires in the frame its call runs.

| Frame | runs | input | on the apex | infers |
|---|---|---|---|---|
| 1 | | `a1`, `b1` | `a1`, `b1` | the step `1 · 1 → out-0, carry-1` |
| 2 | `out-0`, `carry-1` | `a0`, `b1`, `c` expected | the step, covering the digits; `a0`, `b1`, `c` | the step `0 · 1 · c → out-0, carry-1` |
| 3 | `out-0`, `carry-1` | `a1`, `b1`, `c` expected | the step; `a1`, `b1`, `c` | the step `1 · 1 · c → out-1, carry-1` |
| 4 | `out-1`, `carry-1` | `c` expected | the step; `c` | the step `out-1` |
| 5 | `out-1` | | the step | |

The digits that ran in `out`, from the right: `0`, `0`, `1`, `1`. The result is `1100`.

# 6. The same algorithm as code

Written as code, before this design existed, the algorithm was recursive rather than looping. Each of its
parts has a place here:

| in the code                           | in the machine                                                |
|---------------------------------------|---------------------------------------------------------------|
| `addDigits(d1, d2)`, a table of cases | the eight steps, each a case and what was written for it      |
| `getDigit(num, pos)`                  | the environment presenting the next pair                      |
| the `carry` parameter                 | the expected event `c`                                        |
| recursion over `pos`                  | the frame: each pair infers one of the same eight steps       |
| the base case: no digits, no carry    | the frame in which nothing fires, so nothing runs             |
