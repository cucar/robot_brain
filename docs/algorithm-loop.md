# Tap while go, then done, as a machine that could be built

A worked case for [algorithm.md](algorithm.md): a machine that runs `tap` in the frame after every `go` the
world shows, and `done` in the frame after a `stop`, written out neuron by neuron. Nothing here is normative.
It shows what a loop is in this design. There is no counter and no test: the cues are a cycle, each `go` is
answered by a tap because a tap followed every `go`, and the cue that ends the stretch is answered by `done`
because `done` followed it. `while (go) tap(); done();` is two interneurons over the cues and the connections
of four neurons.

---

# 1. The environment

**In.** One channel, laid out over time alone (D1): a base neuron sees the frame before it (D4), and its row
takes the frame after it (D18). It declares one event dimension, `cue`, with two buckets, `go` and `stop`.

**Out.** One action dimension, `act`, with two base actions, `tap` and `done`, in the same channel. One base
action runs per action dimension per position in a frame (D8), and `act` has time alone, so one action runs
per frame.

**Teaching.** The world shows `go` for three frames and `stop` in the fourth, and no cue after that. It runs
`tap` itself in the frame after each `go` and `done` in the frame after the `stop`, rewards every action, and
shows nothing for a while. Then again. Six stretches are enough for everything below. Afterwards the world
shows the cues and runs nothing itself: it executes what the machine outputs, rewards a tap in the frame
after a `go` and `done` in the frame after a `stop`, and pays any other action negatively.

**Policy.** The machine declares its neighborhood policy (D50) with every entry that has an action off in the
past table, and in the future table only event to action and interneuron to action on. Everything below
follows from that: no action is anyone's neighbor, so no function names one and no action enters a history; a
connection reaches base actions only; actions write no connections and never vote; and nothing connects to
an event, so the machine never predicts a cue. It declares its action policy (D51) so that nothing runs where
every candidate's estimate is negative.

**Prices.** A bare activation costs its relations, one relation each (D13), and a call nothing of its own, one
per relation that did not come true and one per argument (D12).

# 2. The neurons

| Kind | Neurons |
|---|---|
| base events | `go`, `stop` |
| base actions | `tap`, `done` |
| the stretch | `{go, go@−1}`, a cycle of one node, `go` after `go` for as long as it runs (D15); its interneuron `gos` |
| the end of the stretch | `{stop, gos@−1}`: the cue after the cycle's call, in its pose of two laps; its interneuron `gos·stop` |

Both functions are over the cues alone, since no action is anyone's neighbor (D50).

# 3. How it was learned

One stretch is four frames of cues: `go` at frames 0, 1 and 2, and `stop` at frame 3. The taps run at frames
1, 2 and 3, and `done` at frame 4.

1. **The stretch is a cycle from the second stretch.** The second `go` sees the first one back, and the third
   sees the second, so `go` after `go` holds twice in a stretch. A function of one relation is worth nothing
   at two occasions, one for its relation and one for its entry (D57, D13), so nothing starts in the first
   stretch. At the second `go` of the second stretch the relation is at strength 3 and starts a function
   (R50): one node with a relation to itself, a cycle of one node (D15). Fit at that frame, it reads back
   round the cycle, one lap: one relation for nothing, taken (R24). Its interneuron `gos` fires there, in a
   pose of one lap (D53). At the third `go` the relation is the call's own function going round, and the call
   takes the lap: two laps. It stands for the three cues, which leave the apex. From the third stretch on the
   cycle's fit is taken at the second `go` in the first election, and the third `go` is a lap.
2. **The end of the stretch is a function over the call.** `stop` sees the third `go` one back. In the first
   stretch that `go` is bare. From the second stretch it stands under the cycle's call, so the relation to
   the call is used in its stead (R50): `stop` after `gos` in its pose of two laps. That holds once a
   stretch, so it is at strength 3 in the fourth stretch, and `{stop, gos@−1}` is a function. Its fit lands
   on `stop` and on the call, and covers the relation: 1 for nothing, taken. Its interneuron `gos·stop` fires
   at the frame of `stop`, and stands for the four cues. `stop` cannot join the cycle: nothing joins a
   function at a later frame than its origin (R50).
3. **What each neuron holds for the frame ahead.** Only an event or an interneuron connects, and only to an
   action (D50).
   - `go`: in the first stretch every `go` stands uncovered through the next frame, where a tap ran, so `go`
     holds `tap` at `+1`, with the reward (R31). From the second stretch on the first `go` is covered at age
     1 and the others at age 0, and they write nothing more.
   - `gos`: its call at the second `go` stands uncovered through the next frame, where a tap ran, and so does
     its call at the third `go` in the second and third stretches. Its connections are keyed by its pose less
     its laps (D25), so `gos` holds `tap` at `+1`, learned once for every length. The call stands for three
     cues, so it stays open two frames (D9): in the second and third stretches, where nothing covers it, it
     also takes the `done` that ran two frames after the third `go`, and `gos` holds `done` at `+2`. From the
     fourth stretch `gos·stop` covers the call at age 1 and it writes nothing more.
   - `stop`: in the first three stretches it stands uncovered through the next frame, where `done` ran, so it
     holds `done` at `+1`. From the fourth stretch it is covered at age 0 and writes nothing more.
   - `gos·stop`: from the fourth stretch it stands uncovered through the next frame, where `done` ran, and
     holds `done` at `+1`. In the frame it first fires it holds nothing, and what it covers speaks for it
     (R48): `stop`, which holds `done`.
   - `tap` and `done` hold nothing: an action writes no connections (D50).

# 4. One stretch, frame by frame, after teaching

| Frame | the world shows | reported as run | voter | infers for the frame ahead | output |
|---|---|---|---|---|---|
| 0 | `go` | | `go` | `tap` at `+1` | `tap` |
| 1 | `go` | `tap` | `gos`, one lap, over two cues | `tap` at `+1` | `tap` |
| 2 | `go` | `tap` | `gos`, two laps, over three cues | `tap` at `+1` | `tap` |
| 3 | `stop` | `tap` | `gos·stop`, over `stop` and the cycle's call | `done` at `+1` | `done` |
| 4 | no cue | `done` | none that holds anything for frame 5 | nothing | nothing |

At each frame one voter stands. The cues it covers are silent (D10), and the action reported in that frame is
no voter (D50). An action the world pays negatively lands on whatever connected to it (R32), so the voter
that proposed it stops proposing it (R47), and where every candidate is negative nothing runs (D51).

# 5. Why nothing false is learned

**An inferred action enters no history.** An action the machine infers stands weakly in its frame (D40). No
action is anyone's neighbor (D50), so it is in no neighborhood and no relation is counted from it, whether it
ran or not. Where the world reports another action in that frame, the report stands and the inferred one is
dropped (R40): in teaching, a tap inferred for the frame after `stop` gives way to the `done` the world ran.

**No cue is ever predicted.** Nothing connects to an event (D50), so no inference names a cue and none fires
weakly. The frame of `done` and the frames after it show no cue, and nothing stands in for one. That is why
`stop` is shown once.

# 6. Another length

Shown `go` for five frames, the machine taps five times. The first `go` stands bare and holds `tap`. From the
second `go` on the cycle's call stands, each later `go` a lap of it, and its connection to `tap` is keyed by
its pose less its laps, so it holds for every length (D25). At the `stop` the function of §2 does not fit: it
names the cycle's call in its pose of two laps, and a pose is part of a call's name (D34). So `stop` stands
bare, and it holds `done` at `+1` from the first three stretches. The cycle's call stands uncovered beside
it at age 1, and holds `done` at `+2`, which names the same frame. Both say `done`, and it runs. A function
over the call at that length forms over its own stretches, as the one above did.

# 7. What this machine does not do

It does not count. The cycle's call at the third `go` infers what it did at the second, so nothing says
"three" from the cues, and nothing ends the stretch but `stop`. That is why this machine is a while loop: the
world ends it, and the machine learns what to do when it does.

# 8. The same thing as code

| in the code | in the machine |
|---|---|
| `while (go) tap();` | `go`'s connection to `tap`, and `gos`'s, held for every length |
| the test `go` | the cue of the frame: a voter infers only in a frame its neuron fired in |
| the loop exiting | `stop`, and `gos·stop` over it and the cycle's call |
| `done();` | their connection to `done` |
| `for (i = 0; i < 3; i++)` | nothing: the design has no counter |
