# Tap until stop, then done, as a machine that could be built

A worked case for [algorithm.md](algorithm.md): a machine that, shown a cue, runs one action in every frame
until the world shows a second cue, and then runs another action once, written out neuron by neuron. Nothing
here is normative, and the machine is shown running: its dictionary and its connections are as listed, and its
histories already hold runs like this one. It shows what a loop is in this design. There is no counter and no
test: the taps are a cycle, the cycle's call infers a tap because a tap followed it, and the cue that ends the
loop stands beside the last tap in a function of its own, which infers what came after it. `while (!stop)
tap(); done();` is three interneurons.

---

# 1. The environment

**In.** One channel, laid out over time alone (D1): a base neuron sees the frame before it (D4), and its row
takes the frame after it (D18). It declares one event dimension, `cue`, with two buckets, `go` and `stop`.

**Out.** One action dimension, `act`, with two base actions, `tap` and `done`, in the same channel, so an
action is a neighbor of the cue and of the action before it (D1). One base action runs per action dimension per
position in a frame (D8), and `act` has time alone, so one action runs per frame.

**Teaching.** The world shows `go`, runs `tap` itself in each of the next three frames, shows `stop` beside
the third tap, runs `done` in the frame after, rewards every action, and shows nothing for a while. Then
again. Three runs are enough for every function below to be in the dictionary. Afterwards the world shows the
cues and runs nothing: it executes what the machine outputs.

**Prices.** A bare activation costs its relations, one relation each (D13), and a call one, one per relation
that did not come true and one per argument (D12).

# 2. The neurons

| Kind | Neurons |
|---|---|
| base events | `go`, `stop` |
| base actions | `tap`, `done` |
| the taps | `{tap, tap@−1}`, a cycle of one node, `tap` after `tap` for as long as it runs (D15); its interneuron `taps` |
| the end of the loop | `{stop, taps@0}`: the cue beside the cycle's call, in its pose of two laps; its interneuron `taps·stop` |
| the exit | `{done, taps·stop@−1}`: `done` after that; its interneuron `taps·stop·done` |

A pair, `{tap, go@−1}`, sits in the dictionary from the second run, never taken (R24), and the exit's call
sees `go` one beyond what it stands for, so the whole run is one more function over that call, which nothing
below turns on.

# 3. How it was learned

1. **The taps are a cycle from the first run.** The second tap sees the first one back, and the third sees the
   second, so at the third tap `tap` after `tap` is at strength 2 and starts a function (R50): one node with
   a relation to itself, a cycle of one node (D15). Placed at that frame, its fit reads back round the cycle,
   two laps, and ends at `go`, which is no tap, without a failure (D54): two relations for one, taken (R24).
   Its interneuron `taps` fires once, at the third tap, in a pose of two laps (D53). It stands for the
   three taps, which leave the apex, and inherits what the fit did not cover: the first tap's relation to `go`,
   now from the call at `−3` (D10). At the second tap of any run the same fit reads one lap, 1 for 1, and is
   not taken: the first two taps stand bare.
2. **The cue does not join the cycle.** `stop` stands beside one tap in three. Its relation to `tap` at offset
   zero is a candidate of the cycle, and the `tap` counts it: `with` one per run, `without` two, so it never
   pays and never joins (D57, R50). `go` before the taps cannot join either: no relation leads on from a node
   of a cycle in the direction the cycle advances in (D15). So the cycle is the taps and nothing else.
3. **The end of the loop is a function over the call.** `stop` fires in the frame `taps` does, at level 0,
   and records the bare third tap beside it at offset zero. When the call lands on that tap, the machine
   changes the entry in `stop`'s row to `taps` in its pose at offset zero (D10), so `stop` holds the call
   beside it in every run. In the second run that relation is at strength 2, and `{stop, taps@0}` is a
   function. Its fit lands on `stop` and on the call, whole, and covers the relation and the call's cost,
   2 for 1, taken. `taps·stop` fires at the next level of the same frame, at that frame, in its pose. The
   cycle's call is covered at age 0, in the frame it fired.
4. **The exit names the end of the loop.** `done` sees `taps·stop` one back. In the third run that relation is
   at strength 2, and `{done, taps·stop@−1}` is a function, 2 for 1 over the call. `taps·stop·done` stands for
   the whole of `tap, tap, tap, stop, done`.
5. **What each voter learns.** `go` stands uncovered through the first tap, which is bare, so it holds `tap`
   at `+1`. The first tap stands uncovered through the second, which is bare, and holds `tap` at `+1`; at
   the third tap it is covered by `taps` and connects once more, to `taps`. The second tap is covered at age 1
   and connects once more, to `taps` (D10). So the base `tap` holds `tap` and `taps` at `+1`, each as strong
   as the runs. `taps` is covered at age 0 by `taps·stop` in every run and writes nothing (D10). `taps·stop`
   stands uncovered through the fourth frame in the first two runs, where `done` stood bare, and is covered
   there by `taps·stop·done` from the third run on, so it holds `done` at `+1` twice and `taps·stop·done` at
   `+1` from every run after. Nothing in those connections is a tap.

# 4. One run, frame by frame, after teaching

| Frame | input | on the apex | infers for the frame ahead | output |
|---|---|---|---|---|
| 0 | `go` | `go` | `tap` at `+1` (R36) | `tap` |
| 1 | `tap` reported as run | the tap, bare | `tap` at `+1`, and `taps` at `+1`, which expands to a tap there (R28) | `tap` |
| 2 | `tap` | the tap, bare | the same | `tap` |
| 3 | `tap`, and `stop` shown | `taps·stop`, over `stop` and the cycle's call, the three taps under it | `done` at `+1`, and `taps·stop·done` at `+1`, which expands to `done` there | `done` |
| 4 | `done` reported as run | `taps·stop·done` | nothing: it has no connection | nothing |

At frames 1 and 2 one voter stands, the bare tap, and both its connections name a tap at the frame ahead. At
frame 3 the voter is `taps·stop`; the cycle's call and the taps under it are covered and silent (D10). The
match reads each inferred call back: the call it names at `0` stands, the relation to it is credited, nothing
is charged, and the inference is proposed (R49). A tap after `stop`, if one ever ran, is paid negatively and
lands on whatever connected to it (R32), so the voter that proposed it stops proposing it (R47).

**Another length.** Taught five taps, the cycle's call stands from the third tap, each later tap's fit lands on
the call before it, 2 for 1, its laps one more, and the call stands uncovered through the next tap, so `taps`
in a pose of two laps holds `taps` in a pose of three at `+1`, a tap. At the fifth tap `stop` shows beside
the call in its pose of four laps, and that is another relation than the one above, since a pose is part of a
call's name (D34): `{stop, taps@0}` over four laps is a second function, with its own interneuron, learned
over its own runs. So the machine taps while nothing but taps has stood, and runs `done` in the frame after
`stop` at every length it was taught, each length through its own exit. One exit for every length is what a
variable over laps will give, and until then the recognition is strict: a run of a length never taught ends
on the bare `stop`, which has no connection to `done`, and the world runs it.

# 5. The stop

**The stop is the cue, in a function with the call.** `stop` is one event among the taps, and the cycle's
call stands beside it. Neither alone says what comes next: the call is covered in that frame and has never
learned anything, and `stop` is a base event that stood once per run. What says `done` is their function,
which has stood before `done` at every run, and speaks because it covers both (D10, R35).

**The cycle's call never infers a stop, and need not.** Where the loop goes on, the call stands uncovered
through the next tap and learns a tap. Where it ends, the call is covered at age 0 by the end's function and
learns nothing. So `taps` holds a tap, or nothing, and never `done`: it cannot end the loop, and the loop is
not its to end. The frame `stop` shows is the frame the end's function fires, and that function holds `done`.

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
`{tap-a, tap-b@0, tap-c@0}` is grouped, two relations for one, and its interneuron `triple` covers them; the
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
