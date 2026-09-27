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
| the cue state | `show`'s pattern, `(ready, one frame ago)`, and its child `shown`, at level 1 |
| the copy program | `again`'s pattern, `(show, two frames ago)` and a parameter `P` over the digit now and the digit two frames ago; its child `copy`, at level 1 |
| value children | `P:0` to `P:9`, one level up, one per digit `P` has passed (D45) |

# 3. How it was learned

1. **The cue becomes a state.** `ready` then `show` recurs on every showing, so `show` holds a pattern for it,
   and its child `shown` fires at level 1 whenever the cues appear. `shown` standing on the apex is the state
   "a copy has been asked for".
2. **The copy becomes a pattern.** At the frame `again` fires, the digit beside it is always the digit shown
   two frames before, whatever that digit is. So in `again`'s history the digit's place now and its place two
   frames back hold one neuron in every row: a parameter (D27, D44). It pays, one value covering two cells, and
   the pattern that names it, with `show` two frames back, is bought; its child `copy` fires at level 1.
3. **The state learns to call the program.** `shown` is still open two frames later, uncovered, and `copy`
   stands on the apex then. So `shown` connects to `copy` at offset two (D25, R31), and every taught showing
   strengthens it.

# 4. One showing, frame by frame, after teaching

| Frame | input | on the apex | what happens |
|---|---|---|---|
| 1 | `ready` | `ready` | |
| 2 | `show`, `7` | `shown`, `7` | `shown` infers `copy` two frames ahead (R36) |
| 3 | | | |
| 4 | `again` | `copy`, and `P:7` beside it | the `7` fires weakly beside `again`, as expected |

In frame 2 the inference places `copy` at its completion, frame 4, and expands it back from there (R28).
`copy`'s body names `show` two frames back, which is frame 2, where `show` stands; and `P` at frame 2 and at
frame 4. Frame 2 has happened, so `P` takes its value from what stands there, the `7`; frame 4 has not, so
expansion places the `7` there, weakly, as an expected event (D44, R40). In frame 4 the world shows `again` and
no digit, the expected `7` stands beside it, `again`'s pattern fits, and `copy` is bought with `P = 7`.

Show a `3` instead and nothing in the machine is different: the same state infers the same program, and the
same parameter carries a `3`. One program serves every digit, because nothing in it names a digit.

# 5. The same thing as code

| in the code | in the machine |
|---|---|
| calling `copy` | `shown` inferring `copy` two frames ahead |
| the function `a => a` | `again`'s pattern, whose parameter spans the digit now and two frames back |
| the parameter `a` | `P`, in `again`'s parameters table |
| binding `a` to the argument | `P` taking its value from the `7` standing at frame 2 |
| `return a` | expansion placing the same value at frame 4 |
| the value | the digit `7`, expected and fired weakly two frames after it was shown |
