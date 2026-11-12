# Tap until stop, then done, as a machine that could be built

A worked case for [algorithm.md](algorithm.md): a machine that, shown a cue, runs one action in every frame
until the world shows a second cue, and then runs another action once, written out neuron by neuron. Nothing
here is normative, and the machine is shown running: its dictionary and its connections are as listed, and its
histories already hold runs like this one. It shows what a loop is in this design. There is no counter and no
test: the taps are a cycle, the cycle's call infers a tap because a tap followed it, and the cue that ends the
loop stands beside the last tap in a function of its own, which infers what came after it. `while (!stop)
tap(); done();` is four interneurons.

---

# 1. The environment

**In.** One channel, laid out over time alone (D1): a base neuron sees the frame before it (D4), and its row
takes the frame after it (D18). It declares one event dimension, `cue`, with two buckets, `go` and `stop`.

**Out.** One action dimension, `act`, with two base actions, `tap` and `done`, in the same channel, so an
action is a neighbor of the cue and of the action before it (D1). One base action runs per action dimension per
position in a frame (D8), and `act` has time alone, so one action runs per frame.

**Teaching.** The world shows `go`, runs `tap` itself in each of the next three frames, shows `stop` beside
the third tap, runs `done` in the frame after, rewards every action, and shows nothing for a while. Then
again. Six runs are enough for every function below to be in the dictionary. Afterwards the world shows the
cues and runs nothing: it executes what the machine outputs.

**Prices.** A bare activation costs its relations, one relation each (D13), and a call nothing of its own, one
per relation that did not come true and one per argument (D12).

# 2. The neurons

| Kind | Neurons |
|---|---|
| base events | `go`, `stop` |
| base actions | `tap`, `done` |
| the start | `{tap, go@−1}`: the first tap after the cue; its interneuron `go·tap` |
| the taps | `{tap, tap@−1}`, a cycle of one node, `tap` after `tap` for as long as it runs (D15); its interneuron `taps` |
| the end of the loop | `{stop, taps@0}`: the cue beside the cycle's call, in its pose of two laps; its interneuron `taps·stop` |
| the exit | `{done, taps·stop@−1}`: `done` after that; its interneuron `taps·stop·done` |

At the second tap of a run the cycle's fit reads the first tap through `go·tap`'s call, which stays where it
stands (D10), so `taps` stands for the taps, and the exit's call for the taps, `stop` and `done`.

# 3. How it was learned

1. **The taps are a cycle from the second run.** The second tap sees the first one back, and the third sees
   the second, so `tap` after `tap` holds twice in a run. A function of one relation is worth nothing at two
   occasions, one for its relation and one for its entry (D57, D13), so nothing starts in the first run. At
   the second tap of the second run the relation is at strength 3 and starts a function (R50): one node with
   a relation to itself, a cycle of one node (D15). Fit at that frame, it reads back round the cycle, one
   lap, and ends at `go`, which is no tap, without a failure (D54): one relation for nothing, taken (R24).
   Its interneuron `taps` fires there, in a pose of one lap (D53). At the third tap the relation is the
   call's own function going round, and the call takes the lap: two laps. It stands for the three taps,
   which leave the apex, and inherits what the fit did not cover: the first tap's relation to `go`, now from
   the call at `−3` (D10). The first tap after the cue holds once a run, so `{tap, go@−1}` is at strength 3
   at the first tap of the third run, and is taken there for nothing: its interneuron `go·tap` fires and
   stands for the cue and the tap. At the second tap the cycle's fit reads one lap, the first tap read
   through that call (D5), 1 for nothing: `taps` fires in a pose of one lap, standing for the two taps, and
   `go·tap`'s call stays on the apex, since `taps` does not name it (D10). At the third tap the relation is
   the call's own function going round, and the call takes the lap: two laps (D53).
2. **The cue does not join the cycle.** `stop` stands beside one tap in three. Its relation to `tap` at offset
   zero is a candidate of the cycle, and the `tap` counts it: `with` one per run, `without` two, so it never
   pays and never joins (D57, R50). `go` before the taps cannot join either: no relation leads on from a node
   of a cycle in the direction the cycle advances in (D15). So the cycle is the taps and nothing else.
3. **The end of the loop is a function over the call.** `stop` fires in the frame `taps` does, at level 0,
   and records the bare third tap beside it at offset zero. When the call lands on that tap, `stop`'s row
   keeps the tap and takes `taps` beside it, in its pose at offset zero (D10), so `stop` holds the call
   beside it in every run from the second. In the fourth run that relation is at strength 3, and
   `{stop, taps@0}` is a function. Its fit lands on `stop` and on the call, and covers the relation: 1 for nothing, taken.
   `taps·stop` fires at the next level of the same frame, at that frame, in its pose. The
   cycle's call is covered at age 0, in the frame it fired.
4. **The exit names the end of the loop.** `done` sees `taps·stop` one back. In the sixth run that relation is
   at strength 3, and `{done, taps·stop@−1}` is a function, 1 for nothing over the call. `taps·stop·done`
   stands for `tap, tap, tap, stop, done`.
5. **What each voter learns.** In the first two runs `go` stands uncovered through the first tap, which is
   bare, so it holds `tap` at `+1`; from the third run it is covered at age 1 by `go·tap` and connects once
   more, to it (D10). `go·tap`'s call stands uncovered through the second tap and connects to `taps` there; `taps`
   at the second tap stands uncovered through the third, where `taps·stop` stands, and connects to it; each
   of those expands to a tap at the frame ahead (R28). At the third tap `taps` is covered at age 0 by
   `taps·stop` in every run and writes nothing there (D10).
   `taps·stop`
   stands uncovered through the fourth frame in its first two runs, the fourth and the fifth, where `done`
   stood bare, and is covered there by `taps·stop·done` from the sixth run on, so it holds `done` at `+1`
   twice and `taps·stop·done` at `+1` from every run after. Nothing in those connections is a tap.

# 4. One run, frame by frame, after teaching

| Frame | input | on the apex | infers for the frame ahead | output |
|---|---|---|---|---|
| 0 | `go` | `go` | `go·tap` at `+1`, which expands to a tap there (R36, R28), and `tap` at `+1` from the first two runs | `tap` |
| 1 | `tap` reported as run | `go·tap`, over the cue and the tap | `taps` at `+1`, a tap there | `tap` |
| 2 | `tap` | `taps`, one lap, over two taps | `taps·stop` at `+1`, a tap there | `tap` |
| 3 | `tap`, and `stop` shown | `taps·stop`, over `stop` and the cycle's call, the three taps under it | `done` at `+1`, and `taps·stop·done` at `+1`, which expands to `done` there | `done` |
| 4 | `done` reported as run | `taps·stop·done` | nothing: it has no connection | nothing |

At frames 0, 1 and 2 one voter stands, and what it names expands to a tap at the frame ahead. At frame 3 the
voter is `taps·stop`; the cycle's call and the taps under it are covered and silent (D10). The
match reads each inferred call back: the call it names at `0` stands, the relation to it is credited, nothing
is charged, and the inference is proposed (R49). A tap after `stop`, if one ever ran, is paid negatively and
lands on whatever connected to it (R32), so the voter that proposed it stops proposing it (R47).

**Another length.** Taught five taps, the cycle's call stands from the second tap, each later tap is a lap of
it, 1 for nothing, and the call stands uncovered through the next tap, so `taps` holds itself at `+1`, a tap,
learned once for every length (D25). At the fifth tap `stop` shows beside
the call in its pose of four laps, and that is another relation than the one above, since a pose is part of a
call's name (D34): `{stop, taps@0}` over four laps is a second function, with its own interneuron, learned
over its own runs. So the machine taps while nothing but taps has stood, and runs `done` in the frame after
`stop` at every length it was taught, each length through its own exit. One exit for every length is what a
variable over laps will give, and until then the recognition is strict: a run of a length never taught ends
on the bare `stop`, which has no connection to `done`, and the world runs it.

# 5. The stop

**The stop is the cue, in a function with the call.** `stop` is one event among the taps, and the cycle's
call stands beside it. Neither alone says what comes next: the call is covered in that frame and has only
learned taps, and `stop` is a base event that stood once per run. What says `done` is their function,
which has stood before `done` at every run, and speaks because it covers both (D10, R35).

**The cycle's call never infers `done`, and need not.** Where the loop goes on, the call stands uncovered
through the next tap and learns what stood there, itself or the end's function over it, each a tap. In the
frame `stop` shows, the call is covered at age 0 by the end's function and learns nothing. So `taps` holds a
tap, and never `done`: it cannot end the loop, and the loop is not its to end. The frame `stop` shows is the
frame the end's function fires, and that function holds `done`.

**Had the world shown nothing.** A connection is to what stood on the apex (D25), and nothing stands in
silence: there is no rest value (§3.3). Without `stop`, the third tap is the cycle's call, uncovered, holding a
tap or nothing, and a neuron with nothing to say does not keep the floor: **speaking falls through the
cover**, so the taps under it speak for it, down to the base (R48), and the base `tap` says tap. The cue
is not optional; it is how the loop ends.

# 6. What this machine does not do

It does not count. The first two taps are one neuron, the bare tap, with one set of connections, and the
cycle's call first stands at the third tap, where a cycle's fit first pays. Nothing on the apex at the second
tap differs from the first, so no function could say "three" from the taps alone, and a loop shorter than
three has no call at all. A count would need a distinct neuron at every tap from the first, which a pair's
call used to give and no pair gives now. That is why this machine is a while loop: the world ends it, and the
machine learns what to do when it does.

# 7. Three in a frame

Inside a frame nothing is back to back. Activations of one frame are at temporal offset zero from one another
and co-occur, with no order among them (D5, D26), and one base action runs per action dimension per position
in a frame (D8). Three taps in one frame are therefore three action dimensions, one channel each (D1): three
tappers, `tap-a`, `tap-b`, `tap-c`, run together. `tap-a` sees the other two at offset zero in every row, so
`{tap-a, tap-b@0, tap-c@0}` is grouped, two relations for nothing, and its interneuron `triple` covers them; the
loop of §4 then runs over triples, `triple` after `triple` a cycle of one node, ended by `stop` the same
way. An order within the frame that mattered would mean the frame is the wrong tick, and the environment
would declare a finer one.

# 8. The same thing as code

| in the code | in the machine |
|---|---|
| `while (!stop) tap();` | the base tap's connections to `tap` and to `taps`, and `taps`'s to itself: a tap after taps |
| `stop` | an event the world shows, which no function of taps holds |
| the loop exiting | `taps·stop`, the cue beside the cycle's call, which was followed by `done` |
| `done();` | `taps·stop`'s connection to `done` and to the function over it |
| `for (i = 0; i < 3; i++)` | nothing: the design has no counter, and the first two taps are one neuron |
| three statements on one line | three action dimensions at offset zero, covered by one neuron |
