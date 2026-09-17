# Addition, as the machine would do it

A worked case for the function model of [algorithm.md](algorithm.md) (D36–D38): binary column addition on a
sheet, first as a machine that could be built by hand, then as one that learns it. Nothing here is normative;
it is the smallest problem that exercises every part of the model, and it is written so that anything the
model cannot express shows up as a gap.

---

# 1. The setup

One event channel, **the sheet**: a grid laid out over rows and columns, whose event dimension has three
buckets, `0`, `1` and blank. Every cell reports every frame, so a blank cell is an activation of the blank
bucket, not silence.

One action dimension with one function, **write**, of shape `(magnitude: digit, thing: cell)` (D36): put this
digit in that cell. Its default is to write nothing.

The environment presents two rows of digits, right-aligned, with an empty answer row below and an empty carry
row above. It rewards a correct digit written in the answer row, and it does nothing else. No `add` exists
anywhere; addition is what the machine does with `write`.

```
carry     ·   ·   1   ·
row 1     ·   1   0   1
row 2     ·   1   1   1
answer    ·   ·   ·   0        ← after the first column
```

# 2. The facts are patterns

A single-column situation is a level-1 event pattern: a digit with a digit at the offset one row below it, and
either a blank or a `1` at the offset one row above. There are eight of them:

| above | row 1 | row 2 | answer bit | carry |
|-------|-------|-------|------------|-------|
| ·     | 0     | 0     | 0          | ·     |
| ·     | 0     | 1     | 1          | ·     |
| ·     | 1     | 0     | 1          | ·     |
| ·     | 1     | 1     | 0          | 1     |
| 1     | 0     | 0     | 1          | ·     |
| 1     | 0     | 1     | 0          | 1     |
| 1     | 1     | 0     | 0          | 1     |
| 1     | 1     | 1     | 1          | 1     |

Each is minted the second time its arrangement occurs, anywhere on the sheet, because a pattern names neighbors
at offsets and D11 makes the column irrelevant. Its child fires wherever that column stands.

# 3. The lessons are connections

Each pattern's child, standing on the apex when the column is resolved, connects to what ran (R31): one call of
`write`, bound as a relation from itself (D25).

```
(write, +1, digit = answer bit, cell = the cell below me)              every pattern
(write, +1, digit = 1,          cell = the cell above-left of me)      the four that carry
```

"The cell below me" is a neighbor relation, `(cell, offset (+1 row, 0))`, so the connection is learned once
and applies in every column. Nothing in it is a coordinate.

# 4. The carry is an event, so there is no program

The four carrying patterns write two cells in one frame: the answer bit below, and a `1` above the next
column. The written `1` is now an event. The next column's situation, "`0` over `1` with a `1` above", is a
different pattern from "`0` over `1`", with its own connections, and it does not match until the carry has been
written beside it. Order comes out of the sheet: a column is resolved when its situation matches, and a column
with a carry pending cannot match until its right neighbor has run. The sheet is the working memory, and the
frame loop is the loop.

Nothing else is needed. No action pattern, no register, no counter. The base case of the recursion is the
frame in which no column matches anything, so nothing fires and it stops.

# 5. The hand-built machine

Everything above can be written down, which is the test that the representation is sufficient:

- eight level-1 event patterns, each a set of neighbors at row offsets;
- one action, `write`, with its shape;
- twelve connections, each a call bound by a relation, at a positive estimate.

Present two rows and it adds them, right to left, one column per frame, with the carry travelling on the sheet.

# 6. The learning machine

The same machine learned rather than built:

1. **A teacher works a few sums.** Every `write` the environment executes appears in the frame as a call with
   its bindings (D37) and sits in the action neuron's history.
2. **The event hierarchy mints the eight patterns** from the columns it sees, as it mints anything that recurs.
3. **Each pattern, on the apex when its column was resolved, connects to the calls that followed**, writing the
   bindings as relations from itself. The lesson pools across columns.
4. **The action hierarchy chunks the carrying pairs.** "Write the answer bit below, write the carry above-left"
   recurs at the same two cells, so the collapse over those calls names both members at offset zero and keeps
   their bindings as relations to one parameter, the column: a level-1 action, *resolve column*, of shape
   `(thing: cell)` (D38).
5. **The teacher stops.** The patterns fire on a new sum, propose the calls they saw follow them, the writes
   run, the reward confirms them, and re-centering settles the table.

Nothing was searched. What the walk (R37) would have had to find by trial, ten candidates per fact in decimal,
two in binary, was shown.

# 7. Decimal is the same

Ten digits and blank instead of two; two hundred facts instead of eight; a walk of up to ten per fact if
learned by trial. The mechanism, the carry on the sheet and the absence of a program are unchanged.
Multiplication is the same with more rows: partial products, then a column addition.

# 8. The recursion, turned inside out

The same algorithm written as code, before this design existed, is recursive rather than looping, and each of
its parts has a place here:

| in the code                                  | in the machine                                      |
|----------------------------------------------|-----------------------------------------------------|
| `addDigits(d1, d2)`, a table of cases        | the eight patterns and their connections            |
| `getDigit(num, pos)`, an adjacent index      | an offset (D6): "the cell at this position from me" |
| the `carry` parameter                        | the written cell in the carry row                   |
| recursion over `pos`                         | the frame loop: one column per frame, the next being whichever matches |
| the base case: no digits, no carry           | the frame in which nothing matches, so nothing fires |

The stack the recursion needed is the sheet, which is why the machine needs no depth, no loop and no memory of
its own.

# 9. What this case does not exercise

- **Exploration from nothing.** The hand-built and the taught machine never explore. A machine left alone with
  a default that writes nothing and rewards that are only ever positive never starts (algorithm-evaluation.md).
- **Two calls, one reward.** A carrying column writes two cells in one frame, and a reward scoped by channel
  and span reaches both alike, so a wrong carry beside a right bit is not told apart from two right writes
  unless the environment pays per cell.
- **Arithmetic without a sheet.** Asked the same sum with nowhere to write, the machine has nowhere to keep the
  carry. That is the hippocampus's job or nothing.
