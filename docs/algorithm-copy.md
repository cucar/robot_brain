# Write the digit you were shown, as a machine that could be built

A worked case for [algorithm.md](algorithm.md): a machine that is shown a digit, is asked, and writes that
digit, written out neuron by neuron. Nothing here is normative, and the machine is shown running: its tables and
connections are as listed, and its histories already hold moments like this one. It shows how the machine
crosses from what it sees to what it does. Nothing relates an event to an action but a connection, so the
crossing is learned, one connection per digit, and what the action brings is expected before it is reported.

---

# 1. The environment

**In.** A digit is shown in event dimension `d`, ten buckets, `0` to `9`. Beside it, in event dimension `cue`,
the world may fire `ask`: write this digit. Both are laid out over time alone, with radius `R = 1` (D1).

**Out.** One action dimension, `write`, with ten base actions, `write-0` to `write-9`. A write that runs
produces its digit: in the frame after `write-7` runs, the world shows a `7` in `d`.

**Teaching.** While the machine is being taught, the world executes the right write itself, in the frame after
each ask, and rewards it (§3.5). Afterwards the machine outputs the write, and the reward says whether it wrote
what was shown.

# 2. The neurons

| Kind | Neurons |
|---|---|
| base events | `ask`, and the ten digits |
| base actions | `write-0` to `write-9` |
| the asked digit | a variable in `ask`'s table, a class of one position beside it whose members are the digits; its value neurons `ask:0` to `ask:9` (D41, D45) |
| the steps | ten functions, one in the table of each write, each with one neighbor: its digit's value neuron a frame before; their children `wrote-0` to `wrote-9` |
| the echoes | ten functions, one in the table of each digit, each with one neighbor: its `wrote` child a frame before |

# 3. How it was learned

1. **The ask and the digit become one neuron.** `ask` sees a digit beside it, at offset zero (D26), and a
   different one each time. An offset that varies is what a variable is seeded on (D47), so `ask`'s table holds
   one there: a class whose members are the digits (D41). A bid covers its bidder (D31), so where the class
   holds, `ask` and the digit are written as one call, and the class's value neuron for that digit, `ask:7`,
   fires at `ask`'s coordinate (D45). No function names a digit there: the position is the variable's (D27). One
   neuron per digit stands on the apex: this digit, asked.
2. **The situation learns the write.** In the taught frames `write-7` ran one frame after `ask:7` and was
   rewarded. `ask:7` was open and uncovered, so it connected to `write-7` at offset one, with the reward
   (R31). That connection is the crossing from the event to the action, and there are ten of them, each learned
   from its own digit's lessons.
3. **The write gets its step.** In `write-7`'s history `ask:7` stands one back in every row, so its table
   holds the function `(ask:7, one back)`, and its child `wrote-7` fires at `write-7`'s coordinate, covering
   both. From then on `ask:7` is covered in the frame the write runs, and in that frame it connects to
   `wrote-7`, the neuron that covered it, with that frame's reward (D10). So it goes on learning what writing a
   `7` earns.
4. **The digit gets its echo.** The frame after, the world shows the `7` that was written. In `7`'s history
   `wrote-7` stands one back on every such frame, so `7`'s table holds `(wrote-7, one back)`, and `wrote-7`
   connects to its child the same way.

# 4. One ask, frame by frame, after teaching

| Frame | input | on the apex | what happens |
|---|---|---|---|
| 1 | `7`, `ask` | `ask:7` | `ask:7` infers `wrote-7` one frame ahead; expanded, that is `write-7`, which is output (R36, R28) |
| 2 | `write-7` reported as run | `wrote-7`, over `ask:7` and `write-7` | `wrote-7` infers the echo; expanded, that is a `7` at frame 3, expected |
| 3 | `7` | the echo's child, over `wrote-7` and the `7` | the `7` was already there, weak, when the world reported it |

In frame 1 `ask:7` reads its connection at offset one and places `wrote-7` at frame 2. Expanding `wrote-7`
places its bidder, `write-7`, at frame 2, and its one neighbor, `ask:7`, at frame 1, where it stands (R28).
`write-7` is a base action at the frame ahead, so it is output. In frame 2 the world reports it as run,
`write-7`'s function holds, and `wrote-7` fires strong. It infers the echo's child one frame ahead, whose
expansion places a `7` at frame 3: a base event, so it fires there weakly, as what the write is expected to
bring (R30, D40). In frame 3 the world shows the `7`, and the report replaces the expectation.

Show a `3` instead and a different neuron stands in frame 1, `ask:3`, with a connection of its own to
`write-3`. Nothing carries the digit across: each digit's write was learned from that digit's own lessons, and a
digit never asked before has no connection yet. That is the difference between this case and a parameter, which
carries whatever stands in it: a parameter holds an event as an event, and only a connection reaches from an
event to an action.

# 5. The same thing as code

| in the code | in the machine |
|---|---|
| `switch (digit)` | which of the class's value neurons stands on the apex |
| `case 7: write(7)` | `ask:7`'s connection to `write-7`, and to `wrote-7` once the step exists |
| the ten cases | ten connections, each learned from its own lessons |
| the return value appearing on the screen | the `7` the world shows in frame 3 |
| knowing what the call will print | the echo, expected a frame before it is reported |
