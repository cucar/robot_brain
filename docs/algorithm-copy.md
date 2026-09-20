# Copy, as a machine that could be built

A worked case for [algorithm.md](algorithm.md): a machine that holds on to whatever digit it was just shown,
`a => a`, written out neuron by neuron. Nothing here is normative, and the machine is shown running: its tables
and connections are as listed, and its histories already hold moments like this one. It shows a value travelling
the whole loop without anything being decided per value: a class neuron binds what was seen, the binding rides
the call, and the call's return names the same class neuron, so what comes back is what went in.

---

# 1. The environment

**In.** A digit is shown in event dimension `d`, ten buckets, `0` to `9`. Every showing is announced: a `ready`
event fires the frame before, and a `show` event fires in the frame the digit does, both in event dimension `cue`.

**Out.** Nothing. The machine acts only on itself.

# 2. The neurons

| Kind | Neurons |
|---|---|
| base events | `ready`, `show`, and the ten digits |
| class neuron | `Digit`, in dimension `d`, whose members are the ten digits (D41) |
| base action | `hold`, in an action dimension the environment does not have, so every call of it is weak (D40) |
| pattern | one, in the table of `show`, with its child `shown` |

# 3. The situation

The pattern in `show`'s table names two neighbors:

| names | offset |
|---|---|
| `ready` | one frame ago |
| `Digit` | now |

`Digit` is present when any digit stands beside `show`, and the activation of `shown` carries which one: its
binding, `Digit = 7` (D38, D41). The pattern pays for its line because it names `ready`: three activations,
`ready`, `show` and the digit, are written as the child and one binding.

# 4. What the situation returns, and what that returns

| Holder | Connection | Reads as |
|---|---|---|
| `shown`, an event neuron | `(hold, one frame on)` (D25) | call `hold` next frame |
| `hold`, an action neuron | `(Digit, the frame it runs)` (D39) | return whatever `Digit` stands for |

Neither connection names a digit. The call of `hold` fires at the coordinate of the `shown` activation that
returned it and carries that activation's bindings (D37), so inside the call `Digit` still stands for `7`. When
`hold` runs, its return names the class neuron, and what fires is the member it is bound to (R40): the digit `7`,
weakly, there for that frame and gone.

# 5. One showing, frame by frame

| Frame | runs | input | on the apex | chooses |
|---|---|---|---|---|
| 1 | | `ready` | `ready` | |
| 2 | | `show`, `7`, and `Digit` bound to `7` | `shown`, `Digit = 7` | `hold`, carrying `Digit = 7` |
| 3 | `hold`, weakly | `7`, weakly, returned by `hold` | `7` | |

In frame 2 the `7` reports `Digit` in `process classes` and the machine adds the `Digit` activation beside it
(§6.6, §7.1), the pattern covers `ready`, `show` and the `7`, and its child stands on the apex. In frame 3 the world shows
nothing, and the `7` is there anyway.

Show a `3` instead and nothing in the machine is different: the same pattern fits, the same two connections
speak, and a `3` comes back. One situation, one call and one return serve every digit, because none of them
ever says which digit it is.

# 6. The same thing as code

| in the code | in the machine |
|---|---|
| the parameter `a` | the class neuron `Digit`, named in the situation's body |
| the argument passed | the binding `Digit = 7`, carried by the activation and then by the call |
| the function | `hold`, called by the situation's connection |
| `return a` | `hold`'s event connection, which names `Digit` |
| the value returned | the digit `7`, fired weakly the frame `hold` runs |
