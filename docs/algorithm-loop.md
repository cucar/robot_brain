# Tap while go, as a machine that could be built

A worked case for [algorithm.md](algorithm.md): a machine that runs one action in the frame after every `go`
the world shows, and nothing in the frame after a `stop`, written out neuron by neuron. Nothing here is
normative. It shows what a loop is in this design when the world drives it. There is no counter and no test:
each `go` is answered by a tap because a tap followed every `go`, and the taps end because no action ever
followed a `stop`. `while (go) tap();` is one function and the connections of the neurons around it.

---

# 1. The environment

**In.** One channel, laid out over time alone (D1): a base neuron sees the frame before it (D4), and its row
takes the frame after it (D18). It declares one event dimension, `cue`, with two buckets, `go` and `stop`.
**The world shows one of the two in every frame.** The cue is never silent.

**Out.** One action dimension, `act`, with one base action, `tap`, in the same channel, so an action is a
neighbor of the cue and of the action before it (D1). One base action runs per action dimension per position
in a frame (D8), and `act` has time alone, so at most one tap runs per frame.

**Teaching.** The world shows `go` for three frames and `stop` for three, and again. It runs `tap` itself in
the frame after each `go` and rewards it, and runs nothing in the frame after a `stop`. Afterwards the world
shows the cues in stretches of any length and runs nothing itself: it executes what the machine outputs,
rewards a tap in the frame after a `go`, and pays a tap in the frame after a `stop` negatively.

**Policy.** Where every candidate's estimate is negative, nothing runs (D51).

**Prices.** A bare activation costs its relations, one relation each (D13), and a call nothing of its own, one
per relation that did not come true and one per argument (D12).

# 2. The first stretch

| Frame | the world shows | runs | relations returned, from the later neuron to the earlier |
|---|---|---|---|
| 0 | `go` | | |
| 1 | `go` | `tap` | `go` after `go`; `tap` after `go`; `tap` beside `go` |
| 2 | `go` | `tap` | `go` after `go`; `go` after `tap`; `tap` after `go`; `tap` after `tap`; `tap` beside `go` |
| 3 | `stop` | `tap` | `stop` after `go`; `stop` after `tap`; `tap` after `go`; `tap` after `tap`; `tap` beside `stop` |
| 4 | `stop` | | `stop` after `stop`; `stop` after `tap` |
| 5 | `stop` | | `stop` after `stop` |

**One relation holds at every tap.** `tap` after `go` holds at frames 1, 2 and 3, since a tap runs in the
frame after every `go`. The `tap` holds a `go` one back in three rows of its history. The `go` holds a tap
one ahead in two rows, and its row of frame 2 takes the tap of frame 3 afterward, which the machine counts
already (R21). So at frame 3 the relation is at strength 3, and grouping starts a function for it (R50):
`{tap, go@−1}`, two nodes, origin `tap` (D15).

| | relations |
|---|---|
| A, the relation between the two | 1 |
| B, the call's cost: nothing | 0 |
| savings per occasion | 1 |
| in the dictionary: one relation and its entry | 2 |
| worth at `n` occurrences | `n − 2` |

Fit before the second election of frame 3, it covers that one relation for nothing and is taken (R24). Its
interneuron, call it `go·tap`, fires at the tap of frame 3 and stands for that tap and the `go` before it
(D53). No other relation of the stretch has held three times.

**Every tap from then on stands under a call of `go·tap`.** A tap never runs without a `go` one back, so at
every later tap the function fits, 1 for nothing, and is taken in the first election of its frame (R24). The
tap is covered at age 0, in the frame it fires, and the `go` before it at age 1 (D10).

**The stretches recur too.** `stop` after `stop` holds twice in every stretch of three stops, and one call of
`go·tap` stands one frame after another twice in every stretch of three taps. Each starts a function at its
own strength of 3 (R50). What covers a `go`, a `stop` or a call from then on has stood before the same thing
as what it covers, so it comes to hold the same at `+1`, and until it has learned that, what it covers speaks
for it (R48). None of them changes what is output.

# 3. What each neuron holds for the frame ahead

**`go`.** At frames 0 and 1 the `go` stands uncovered through the next frame, where a tap ran, so `go` holds
`tap` at `+1`, with the reward (R31), and `go` at `+1` beside it. From frame 2 on every `go` is covered at
age 1 by `go·tap`'s call, and connects once more, to it (D10): `go` holds `go·tap` at `+1`, which expands to a
tap at the frame ahead and a `go` at this one (R28).

**`go·tap`.** Its call at the first or the second tap of a stretch stands uncovered through the next frame,
where the next call of `go·tap` stands beside the next `go`, so it holds `go·tap` at `+1` and `go` at `+1`.
Its call at the last tap stands through a frame of `stop`, so it holds `stop` at `+1`.

**`stop`.** A `stop` stands uncovered through the next frame, where a `stop` or a `go` stood and nothing ran.
It holds `stop` at `+1` and `go` at `+1`, and no action.

**`tap`.** The taps of frames 1 and 2 stood bare, so `tap` holds a little: `tap` at `+1` from frame 1. Every
tap from frame 3 on is covered at age 0, writes nothing but its connection to `go·tap` (D10), and is not
heard: its coverer speaks (R46).

# 4. A stretch after teaching, frame by frame

| Frame | the world shows | reported as run | voters | infers for the frame ahead | output |
|---|---|---|---|---|---|
| 0 | `go` | | `go` | `tap` at `+1`; `go·tap` at `+1`, a tap there, its `go` standing at this frame | `tap` |
| 1 | `go` | `tap` | `go`; `go·tap`, over the `go` of 0 and the tap | the same from `go`; `go·tap` at `+1` from the call, its `go` standing | `tap` |
| 2 | `go` | `tap` | `go`; `go·tap` | the same | `tap` |
| 3 | `stop` | `tap` | `stop`; `go·tap` | `stop`: no action. The call: `go·tap` at `+1`, struck | nothing |
| 4 | `stop` | | `stop` | no action | nothing |

At frames 0, 1 and 2 the `go` of the frame is a voter, and what it names expands to a tap at the frame ahead.
The match reads each inferred call back (R49): `go·tap` at `+1` has its `go` at the present frame. Where a
`go` stands there, the relation would come true and the tap is proposed. At frame 3 a `stop` stands there, the
function would not pay, and the match strikes the inference: nothing names a tap for frame 4, and nothing
runs. A tap in the frame after a `stop`, if one ever ran, is paid negatively and lands on whatever connected
to it (R32); its estimate turns negative, and where every candidate is negative nothing runs (D51).

# 5. The cue is never silent

Each voter also infers a cue for the frame ahead: `go` holds `go` at `+1`, `go·tap` holds `go` and `stop`,
and `stop` holds `stop` and `go`. The cue with the largest share of voters is output, weakly, in that frame
(R47, R40). At frame 2 that is `go`, for frame 3. The world shows `stop` there, and where the report names a
dimension the report stands and the weak activation is dropped (R40). The world names the cue in every frame,
so every cue the machine infers is confirmed or dropped in the frame it was inferred for, and none enters a
history as something seen.

**Had the cue been silent.** In a frame where the world shows no cue, an inferred cue is what the machine had
in that frame (D40): it enters the histories of everything beside it and counts as seen. A `go` the machine
inferred would be answered by a tap like one the world showed, and the machine would run on what it expected.
The cue in every frame is what keeps the stretch the world's.

# 6. Another length

Shown `go` for five frames, the machine taps five times: every frame of `go` is frame 1 of the table above,
with the same voters and the same inferences, and the fifth tap runs in the frame the first `stop` shows.
Shown `stop` for any number of frames it runs nothing. Nothing in what decides the output holds a length.

# 7. What this machine does not do

It does not count. What stands on the apex at the third `go` infers what the apex at the second did, so no
function says "three" from the taps, and nothing ends the stretch but the cue. That is why this machine is a
while loop: the world ends it, and the machine answers each frame as it comes.

# 8. Three in a frame

Inside a frame nothing is back to back. Activations of one frame are at temporal offset zero from one another
and co-occur, with no order among them (D5, D26), and one base action runs per action dimension per position
in a frame (D8). Three taps in one frame are therefore three action dimensions, one channel each (D1): three
tappers, `tap-a`, `tap-b`, `tap-c`, run together. `tap-a` sees the other two at offset zero in every row, so
`{tap-a, tap-b@0, tap-c@0}` is grouped, two relations for nothing, and its interneuron `triple` covers them;
the loop of §4 then runs over triples, `triple` after `go` as `tap` was. An order within the frame that
mattered would mean the frame is the wrong tick, and the environment would declare a finer one.

# 9. The same thing as code

| in the code | in the machine |
|---|---|
| `while (go) tap();` | `go`'s connections to `tap` and to `go·tap`, and `go·tap`'s to itself: a tap after every `go` |
| the test `go` | the cue of the frame: the match strikes a tap inferred through `go·tap` where no `go` stands |
| the loop exiting | `stop`, which holds no action |
| `for (i = 0; i < 3; i++)` | nothing: the design has no counter |
| three statements on one line | three action dimensions at offset zero, covered by one neuron |
