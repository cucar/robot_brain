# Addition, as the machine would do it

A worked case for the function model of [algorithm.md](algorithm.md) (D37, D38): binary column addition on a
sheet, first as a machine that could be built by hand, then as one that learns it. Nothing here is normative;
it is the smallest problem that exercises every part of the model, and it is written so that anything the
model cannot express shows up as a gap.

---

# 1. The setup

**The sheet** is an event channel: a grid laid out over rows and columns, whose event dimension has three
buckets, `0`, `1` and blank. Every cell reports every frame, so a blank cell is an activation of the blank
bucket, not silence. Four rows: a carry row, the two numbers `a` and `b`, and an answer row.

**The focus** is an event channel on the same layout with one bucket, `here`, and one activation: the cell the
machine's actions apply to. The environment holds it; the machine only sees it.

**The actions** are one action dimension of six base actions, none with an argument (D37): `up`, `down`,
`left`, `right`, `write-0`, `write-1`. The first four move the focus; the last two write into the cell it is on.

The environment presents two numbers, right-aligned, with the focus on the rightmost digit of `a`. It rewards a
correct digit written in the answer row, and it does nothing else. No `add` exists anywhere.

```
carry     ·   ·   ·   ·
a         ·   1   0  [1]        [ ] the focus
b         ·   1   1   1
answer    ·   ·   ·   ·
```

# 2. Every lesson is one step

A situation's activation is open for as many frames as its level reaches (D9), two at level 1, and it connects
only to what runs while it is open (R31). So a situation at the top of a column cannot hold "resolve this
column", which takes six frames or eleven; it can hold "the next step from here". Addition is therefore a chain
of one-step lessons, each returned by the situation the previous step created, and what carries a column's
identity from the frame it was seen to the frame it is needed is not a connection but a pattern: neighborhoods
include earlier frames (D5), so "the focus is on a blank answer cell, and two frames ago and two rows up the
column was `1` over `1`" is an ordinary pattern, and its lesson is `write-0`.

The path has to be laid so that every decision's facts are within reach of the situation that makes it. One
that does, with every fact within two cells and two frames:

```
a column that does not carry     down, down, write-s, left, up, up
a column that carries            up, left, write-1, right, down, down, down, write-s, left, up, up
```

Both end with the focus on `a` of the next column. The carry is written first, because from `a` the next
column's carry cell is one step up and one left, in view, and a situation that sees it already holds a `1`
knows its carry is done.

# 3. The situations and what they return

| The focus is on | and the situation also sees | returns |
|---|---|---|
| a digit of `a` | a column that carries, and a blank up-left | `up` |
| a digit of `a` | a column that does not carry, or a `1` up-left | `down` |
| the carry row | one frame ago, one row down, a carrying column | `left` |
| a blank carry cell | one frame ago, the focus one cell right | `write-1` |
| a carry cell holding `1` | it was written one frame ago | `right` |
| the carry row | one frame ago, the focus one cell left | `down` |
| a digit of `b` | one frame ago, the focus one row up | `down` |
| a blank answer cell | two frames ago, two rows up, column situation `S` | `write-s`, the answer bit of `S` |
| an answer cell just written | | `left` |
| a blank answer cell | a written cell one to the right, written two frames ago | `up` |
| a digit of `b` | one frame ago, the focus one row down | `up` |

Eight column situations `S`, carry above or not, `a`, `b`, give eight rows of the eighth kind. The rest do not
depend on the digits at all. Each is minted the second time its arrangement occurs, in any column, because a
pattern names neighbors at offsets and D11 makes the column irrelevant.

# 4. The carry is an event, so there is no loop

A carrying column writes a `1` above the next column before it does anything else. That `1` is an event, so the
next column's situation is one of the four with a carry above, a different pattern with a different answer bit.
When a column is done the focus is on the next one, a situation fires, and its step runs. The loop is the frame,
the counter is the focus, the carry register is the sheet, and the base case is the frame in which the focus is
on blank over blank with nothing above, which matches nothing, so nothing runs and it stops.

# 5. The functions are compression, and callable only from high enough

The action hierarchy sees the same stream the teacher produced and chunks it as it chunks anything (D38):

```
column(X)         down, down,  X,  left, up, up
carry             up, left, write-1, right, down
```

Over the columns that do not carry, every member agrees but the third, `write-0` in some and `write-1` in
others: the collapse names the five that agree and keeps the third as a hole (D27). One hole, one parameter.
These are in the file, where they shorten the action stream, and they stand on the apex of the action
hierarchy. What they are not is something a level-1 situation calls: a voter can start a program only from an
offset at least as far out as the program is long (R36). A situation whose window holds six frames, level 3 and
up, can call `column(X)` whole, with its argument; below that, the same behavior is dispatched a step at a
time. **A learned function is a compression first, and a callable unit only for a situation tall enough to see
its whole length.**

# 6. The hand-built machine

Everything above can be written down, which is the test that the representation is sufficient: the event
patterns of §3, and one connection from each to the base action it returns, at a positive estimate. Present two
numbers and it adds them, right to left. Building it is also how the path of §2 gets checked: a step whose facts
turn out to be out of reach of its situation shows up as a pattern that cannot be written.

# 7. The learning machine

1. **A teacher works a few sums.** Every base action the environment executes appears in the frame as a call
   (D37) and sits in its action neuron's history.
2. **The event hierarchy mints the situations of §3** from what it sees, the focus included, the recent past
   included.
3. **Each situation, on the apex when the next step ran, connects to it** at offset one. The lesson pools
   across columns.
4. **The action hierarchy chunks the teacher's stream** into `column(X)` and `carry`, with the hole where the
   digit varied.
5. **The teacher stops.** The situations fire on a new sum and return the steps they saw follow them, the
   reward confirms the writes, and re-centering settles the table.

Nothing was searched. What a walk would have had to find by trial was shown.

# 8. Decimal is the same

Ten digits and blank; two hundred column situations; ten `write` actions; the same path and the same one hole.
Multiplication is the same with more rows: partial products, then a column addition.

# 9. The recursion, turned inside out

The same algorithm written as code, before this design existed, is recursive rather than looping, and each of
its parts has a place here:

| in the code                             | in the machine                                                   |
|-----------------------------------------|------------------------------------------------------------------|
| `addDigits(d1, d2)`, a table of cases   | the eight column situations and the bit each returns             |
| `getDigit(num, pos)`, an adjacent index | an offset (D6), and the focus moving one cell                    |
| the `carry` parameter                   | the `1` written in the carry row                                 |
| recursion over `pos`                    | the frame loop: a column ends on the next, whose situation fires |
| the base case: no digits, no carry      | the frame in which nothing matches, so nothing runs              |

The stack the recursion needed is the sheet, which is why the machine needs no depth, no loop and no memory of
its own.

# 10. What this case shows is missing

- **The path is part of the problem.** The steps work only because every decision's facts are within reach of
  the situation that makes it, and the path was laid out by hand to make that so. A teacher who walks a
  different path may teach steps the machine cannot represent at the level they occur. Nothing in the design
  says how a machine finds a path with that property.
- **A situation that recurs unchanged runs again.** If a step leaves what its situation sees exactly as it was,
  the same situation fires and the same step runs, forever. Here the carry is written first so that the second
  visit to `a` sees a `1` up-left; in general only a negative reward and the walk break such a loop.
- **Families of actions.** "Write the bit this column gives" is eight lessons in binary. In decimal it is two
  hundred, and in general it is a mapping from event neurons to action neurons that the design has no object
  for; a situation with a hole, handing its filler to the action it returns, is where that would go.
- **Exploration from nothing is not exercised.** The hand-built and the taught machine never explore. The
  machine explores only when an estimate turns negative (R37), by design: it is deterministic and it does not
  look further while things are good enough. An environment that wants addition discovered rather than taught
  has to make not adding hurt.
- **Arithmetic without a sheet.** Asked the same sum with nowhere to write, the machine has nowhere to keep the
  carry. That is the hippocampus's job or nothing.
