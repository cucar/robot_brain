# Copy, as a machine that could be built

A worked case for [algorithm.md](algorithm.md): a machine that gives back whatever digit it was just shown,
`a => a`, written out neuron by neuron. Nothing here is normative, and the machine is shown running: its tables
and connections are as listed, and its histories already hold moments like this one. It shows a value carried
forward in time by a parameter, with nothing deciding anything per value: a cue puts the machine in a state,
the state infers the copy program, and the program's parameter takes its value from what stands now and places
it two frames on.

---

# 1. The environment

**In.** A digit is shown in event dimension `d`, ten buckets, `0` to `9`. Every showing is announced in event
dimension `cue`: `ready` fires the frame before the digit, `show` fires with it, and `again` fires two frames
after it. While the machine is being taught, the world shows the same digit again with `again`. Afterwards it
shows `again` alone, and the digit is the machine's to supply.

**Out.** Nothing. The machine acts only on itself.

# 2. The neurons

| Kind | Neurons |
|---|---|
| base events | `ready`, `show`, `again`, and the ten digits |
| the cue state | `show`'s constant, `(show, 0)`, `(ready, one frame ago)`, and its child `shown`, at level 1 |
| the cue pair | `again`'s constant, `(again, 0)`, `(show, two frames ago)`, and its child `asked`, at level 1 |
| the parameter | `P`, in `again`'s table, over the digit now and the digit two frames ago; its child and its value neurons `P:0` to `P:9`, at level 1 (D45) |
| the copy program | a constant at level 1 over `asked` and `P`'s child, both at `again`'s coordinate; its child `copy`, at level 2 |

# 3. How it was learned

1. **The cue becomes a state.** `ready` then `show` recurs on every showing, so `show` holds a constant for it,
   and its child `shown` fires at level 1 whenever the cues appear. `shown` standing on the apex is the state
   "a copy has been asked for".
2. **The cue pair becomes a constant.** `show` two frames before `again` recurs on every showing, so `again`
   holds a constant for it, and its child `asked` fires at `again`'s coordinate.
3. **The digit becomes a parameter.** At the frame `again` fires, the digit beside it is always the digit shown
   two frames before, whatever that digit is. So in `again`'s history the digit's place now and its place two
   frames back hold one neuron in every row: a parameter, found by its own collapse (D27, D44). It pays, one value
   covering two cells, and when its bid is accepted its child and its value neuron for that digit fire at
   `again`'s coordinate, beside `asked`.
4. **The copy becomes a pattern, one level up.** `asked` and `P`'s child stand together at `again`'s coordinate
   on every showing, and the value neuron beside them varies. A constant over the two is built in `asked`'s table,
   and its child `copy` stands at level 2 (D2). It names no digit and no value neuron.
5. **The state learns to call the program.** `shown` is still open two frames later, uncovered, and `copy`
   stands on the apex then. So `shown` connects to `copy` at offset two (D25, R31), and every taught showing
   strengthens it.

# 4. One showing, frame by frame, after teaching

| Frame | input | on the apex | what happens |
|---|---|---|---|
| 1 | `ready` | `ready` | |
| 2 | `show`, `7` | `shown`, `7` | `shown` infers `copy` two frames ahead (R36) |
| 3 | | | |
| 4 | `again` | `copy`, over `asked` and `P`'s child; `P:7` beside them | the `7` fires weakly beside `again`, as expected |

In frame 2 the inference places `copy` at its completion, frame 4, and expands it back from there (R28). `copy`'s
cells are `asked` and `P`'s child, both at frame 4. `asked` expands to `again` at frame 4 and `show` at frame 2,
where `show` stands. `P`'s child expands to the parameter, the digit at frame 2 and at frame 4: frame 2 has
happened, so `P` takes its value from what stands there, the `7`; frame 4 has not, so expansion places the `7`
there, weakly, as an expected event (D44, R40). In frame 4 the world shows `again` and no digit, the expected `7`
stands beside it, `again`'s constant and `P` both fit, their children fire with `P:7`, and `copy` is recognized
over them.

Show a `3` instead and nothing in the machine is different: the same state infers the same program, and the
same parameter carries a `3`. One program serves every digit, because nothing in it names a digit.

# 5. The same thing as code

| in the code | in the machine |
|---|---|
| calling `copy` | `shown` inferring `copy` two frames ahead |
| the function `a => a` | `copy`, a constant over the cue pair's child and the parameter's |
| the parameter `a` | `P`, in `again`'s table, over the digit now and two frames back |
| binding `a` to the argument | `P` taking its value from the `7` standing at frame 2 |
| `return a` | expansion placing the same value at frame 4 |
| the value | the digit `7`, expected and fired weakly two frames after it was shown |
