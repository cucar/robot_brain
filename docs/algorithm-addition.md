# Addition, as a machine that could be built

A worked case for [algorithm.md](algorithm.md): a machine that adds two binary numbers, written out neuron by
neuron. Nothing here is normative, and the machine is shown running: its tables and connections are as listed,
and its histories already hold sums like this one. It shows the parts of the design doing one small job: a step
per case, naming the digits it saw and the actions it ran; a connection from the digits to the step; and an
action nothing outside the machine sees, the carry, read back a frame later as the machine's only variable.

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
does not have that dimension, so `carry-1` runs and nothing outside the machine sees it (D37). It stands in the
frame it runs like any action, and what was done a frame ago is context as much as what was seen (D5), so the
step of the next column names `carry-1` a frame before, beside that column's digits. Writing the carry is
running `carry-1`; reading it is recognizing that `carry-1` ran. That is the whole of the variable, and nothing
had to be shown to the machine to make it one.

# 3. The neurons

| Kind | Neurons |
|---|---|
| base events | `a0`, `a1`, `b0`, `b1` |
| base actions | `out-0`, `out-1`, `carry-1` |
| patterns | nine steps, in the tables of `out-0` and `out-1`, each naming what stood and ran the frame before and the actions of its own, each with a child |

# 4. The steps, and what each infers

A step is a pattern in the table of the `out` digit that ran, naming the pair of digits a frame before it, and
`carry-1` there if it ran, and `carry-1` beside it when it runs (D5). Matched, it says this case happened and
this was written for it; inferred, it writes the digit and, where it names `carry-1` beside it, carries (R30).

| Step, in the table of | names a frame before | names beside it |
|---|---|---|
| `out-0` | `a0`, `b0` | |
| `out-1` | `a0`, `b1` | |
| `out-1` | `a1`, `b0` | |
| `out-0` | `a1`, `b1` | `carry-1` |
| `out-1` | `a0`, `b0`, `carry-1` | |
| `out-0` | `a0`, `b1`, `carry-1` | `carry-1` |
| `out-0` | `a1`, `b0`, `carry-1` | `carry-1` |
| `out-1` | `a1`, `b1`, `carry-1` | `carry-1` |
| `out-1` | `carry-1` | |

What a frame infers is its connections (D25): what stood uncovered on the apex connected, a frame on, to the
step that followed it, and that step's estimate is what the sums earned. The step is expanded (R28): `out-0` or
`out-1` runs, and `carry-1` runs where the step names it beside it. After a carry, the step that carried stands
on the apex beside the next pair of digits. The pair alone connects both to the step that names `carry-1` and to
the one that does not; the step that carried connects only to steps that name it, since those are what followed
it on every sum, so the two together infer the step that reads the carry.

**The last carry.** In the frame after the numbers run out no digits arrive, and only the step that carried
stands on the apex. On every taught sum whose last column carried, what followed it there was the last step of
the table, which names `carry-1` a frame before and nothing else, so its connection speaks and the final `1` is
written.

# 5. One sum, frame by frame

`101 + 111`. The pairs, from the right, are `(1, 1)`, `(0, 1)`, `(1, 1)`. What is chosen in one frame runs in the
next (R29).

| Frame | runs | input | on the apex | infers |
|---|---|---|---|---|
| 1 | | `a1`, `b1` | `a1`, `b1` | the step `1 · 1 → out-0, carry-1` |
| 2 | `out-0`, `carry-1` | `a0`, `b1` | the step, covering the digits before and what ran; `a0`, `b1` | the step `0 · 1 · carry → out-0, carry-1` |
| 3 | `out-0`, `carry-1` | `a1`, `b1` | the step; `a1`, `b1` | the step `1 · 1 · carry → out-1, carry-1` |
| 4 | `out-1`, `carry-1` | | the step | the step `carry → out-1` |
| 5 | `out-1` | | the step | |

The digits that ran in `out`, from the right: `0`, `0`, `1`, `1`. The result is `1100`, a sum the machine need
never have seen: every column is one of the nine steps, and the carry takes each column's outcome into the next.

# 6. The same algorithm as code

Written as code, before this design existed, the algorithm was recursive rather than looping. Each of its
parts has a place here:

| in the code                           | in the machine                                                |
|---------------------------------------|---------------------------------------------------------------|
| `addDigits(d1, d2)`, a table of cases | the nine steps, each a case and what was written for it       |
| `getDigit(num, pos)`                  | the environment presenting the next pair                      |
| the `carry` parameter                 | `carry-1`, run in one frame and recognized in the next        |
| recursion over `pos`                  | the frame: each pair infers one of the same steps             |
| the base case: no digits, no carry    | the frame in which nothing fires, so nothing runs             |
