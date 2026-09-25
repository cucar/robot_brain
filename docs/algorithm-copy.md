# Copy, as a machine that could be built

A worked case for [algorithm.md](algorithm.md): a machine that holds on to whatever digit it was just shown,
`a => a`, written out neuron by neuron. Nothing here is normative, and the machine is shown running: its tables
and connections are as listed, and its histories already hold moments like this one. It shows a value travelling
the whole loop without anything being decided per value: a variable holds what was seen, the step that follows
expects that variable's value, and the variable is still standing when the expectation reads it, so what comes
back is what went in.

---

# 1. The environment

**In.** A digit is shown in event dimension `d`, ten buckets, `0` to `9`. Every showing is announced: a `ready`
event fires the frame before, and a `show` event fires in the frame the digit does, both in event dimension `cue`.

**Out.** Nothing. The machine acts only on itself.

# 2. The neurons

| Kind | Neurons |
|---|---|
| base events | `ready`, `show`, and the ten digits |
| class neuron | `Digit`, a variable that holds whichever digit was shown (D41) |
| base action | `hold`, in an action dimension the environment does not have, so nothing outside the machine sees it run (D37) |
| patterns | one in the table of `show`, with its child `shown`; one in the table of `hold`, the step, with its child `held` |

# 3. The situation

The pattern in `show`'s table names two neighbors:

| names | offset |
|---|---|
| `ready` | one frame ago |
| `Digit` | now |

`Digit` is fit by any digit standing beside `show` (D38), and when the bid is accepted a `Digit` variable
fires one level up at the digit's coordinate, holding `7` (§7.4). The pattern pays for its line because it
names `ready`: three activations, `ready`, `show` and the digit, are written as the child and one variable.
`Digit` itself came from the collapse of `show`'s history (D27): the spot beside `show`, a frame after `ready`,
was filled every time by a different digit. On its own that offset would be left out, one value for one
neighbor; it joins the candidate through the constants `ready` and `show`, whose rows it stands in, and it is
worth having not for what it saves but for what it exposes: the digit, as a variable one level up.

# 4. What the situation infers, and what that expects

The step, in `hold`'s table, names `show` and `Digit` a frame before and `hold` beside it (D5); its child is
`held`.

| Holder | Connection | Reads as |
|---|---|---|
| `shown` | `(held, one frame on)` (D25) | infer the step next frame: run `hold` |
| `held` | `(Digit, one frame on)` (D25) | expect whatever `Digit` holds the frame after |

Neither connection names a digit. The step is inferred at the coordinate of the `shown` activation (D37), one
frame after the `Digit` variable fired, and the variable is open for its window (D9), so it is still standing
within reach. `held`'s connection was learned while the world kept showing the digit a second time; now the
expectation stands in for it (R40): it names `Digit`, and what fires is the value of the variable standing
there: the digit `7`, weakly, there for that frame and gone.

# 5. One showing, frame by frame

| Frame | runs | input | on the apex | infers |
|---|---|---|---|---|
| 1 | | `ready` | `ready` | |
| 2 | | `show`, `7` | `shown`, and `Digit` holding `7` | the step `held` |
| 3 | `hold` | | `held`, covering `show`, `7` and `hold` | `Digit`'s value, expected |
| 4 | | `7`, weakly, expected | `7` | |

In frame 2 the pattern covers `ready`, `show` and the `7`, and its child stands on the apex beside a `Digit`
variable holding `7` (§7.4). In frame 3 `hold` runs and the
step is matched. In frame 4 the world shows nothing, and the `7` is there anyway.

Show a `3` instead and nothing in the machine is different: the same pattern fits, the same two connections
speak, and a `3` comes back. One situation, one step and one expectation serve every digit, because none of them
ever says which digit it is.

# 6. The same thing as code

| in the code | in the machine |
|---|---|
| the variable `a` | the class neuron `Digit`, named in the situation's body |
| the value `a` holds | the `Digit` variable on the level, holding `7` |
| the function | the step `held`, inferred by the situation's connection |
| `return a` | `held`'s connection, which names `Digit` and reads the variable standing there |
| the value | the digit `7`, expected and fired weakly the frame after `hold` runs |
