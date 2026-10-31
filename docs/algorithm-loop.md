# Tap three times and stop, as a machine that could be built

A worked case for [algorithm.md](algorithm.md): a machine that, shown a cue, runs one action in each of the
next three frames and then runs nothing, written out neuron by neuron. Nothing here is normative, and the
machine is shown running: its dictionary and its connections are as listed, and its histories already hold
runs like this one. It shows what a loop is in this design. There is no counter: the count is an interneuron,
the loop variable is which interneuron stands on the apex, and the loop ends because the last one has never
been followed by the action. The same machine, taught to stop on a cue instead of a count, is a while loop,
and that is the one the design runs natively.

---

# 1. The environment

**In.** One channel, laid out over time alone (D1): a base neuron sees the frame before it (D4). It declares
one event dimension, `cue`, with two buckets, `go` and `done`.

**Out.** One action dimension, `tap`, with one base action, `tap`, in the same channel, so a tap is a neighbor
of the cue and of the tap before it (D1). One base action runs per action dimension per position in a frame
(D8), and `tap` has time alone, so a tap is one per frame: what the loop counts is frames.

**Teaching.** The world shows `go`, runs `tap` itself in each of the next three frames and rewards each,
shows `done` in the fourth, and shows nothing for a while. Then again. Five runs are enough for every function
below to be in the dictionary.

**Prices.** The layout has no spatial dimension, so a bare record is two numbers, the neuron and the frame
(D13), and a call's record is two, plus one per relation that did not come true and one per value (D12).

# 2. The neurons

| Kind | Neurons |
|---|---|
| base events | `go`, `done` |
| base action | `tap` |
| the count | three functions, each naming `tap` and one neuron a frame back: `{tap, go@−1}`, `{tap, go·tap@−1}`, `{tap, go·tap²@−1}`; their interneurons `go·tap`, `go·tap²` and `go·tap³` |
| the end | `{done, go·tap³@−1}`; its interneuron `go·tap³·done` |

That is the whole machine: three neurons for the three states of the loop, and one for its end.

# 3. How it was learned

1. **The first tap names the cue.** In the second run, the tap after `go` sees `go` at `−1`, and `tap`'s
   activate call returns that relation at strength 2 (R21). Grouping makes `{tap, go@−1}` of it (R50). It
   writes two records, four numbers, as one call of two, and its entry is one relation and two neurons, three,
   so its worth is positive at strength 2 (D57). The election takes it (R24), it is allocated `go·tap` (R16),
   and `go·tap` fires at the tap's frame, the latest of what it covers (D53), at the frame's second level.
2. **The second tap names the first interneuron.** Once `go·tap` stands, the next tap sees it at `−1`, since
   a neighbor is whatever stands on the apex within reach (D5), and nothing under it (D10). In the third run
   the relation to `go·tap` is at strength 2, and `{tap, go·tap@−1}` is grouped: it covers the tap and
   `go·tap`'s record, four numbers, for two (D22). Its interneuron
   `go·tap²` covers `go·tap` and the tap, three frames of run.
3. **The third tap names the second.** `{tap, go·tap²@−1}` in the fourth run, the same way, and `go·tap³`.
   Each state is one frame longer than the one before it. The chain grows one state per run, because an
   interneuron has to stand before the tap after it can see it. Nothing closes on itself: the index is searched
   by what an interneuron would stand for, read back from where the call would stand (D36, R43), and read from
   there `go·tap³` is `tap, tap, tap, go` where `go·tap²` is `tap, tap, go`, so each length is its own
   interneuron.
4. **The end names the last state.** `done` sees `go·tap³` at `−1` in every run, and in the fifth `{done,
   go·tap³@−1}` is grouped; `go·tap³·done` covers both. Nothing follows it within reach but silence, so it
   holds no connection, and needs none.
5. **Every state learns what came next.** Each interneuron stood open and uncovered in the frame after it
   fired, and in that frame something covered it: `go` by `go·tap`, `go·tap` by `go·tap²`, `go·tap²` by
   `go·tap³`, and `go·tap³` by `go·tap³·done`. An activation covered in a frame connects once, to the neuron
   that covered it, with that frame's reward (D10). So each state holds one connection at offset `+1`, to the
   state after it, and the first three carry the tap's reward. From the runs before the next state existed,
   each also holds the bare tap, or `done`, at `+1`, from the frames where that stood uncovered; those
   connections stay (R31) and place the same thing.

Nothing in `tap`'s dictionary entries says three. `tap` has three functions because three different things
have stood before a tap, and the interneurons built from them are as long as the runs the world showed.

# 4. One run, frame by frame, after teaching

| Frame | input | on the apex | infers for the frame ahead | output |
|---|---|---|---|---|
| 0 | `go` | `go` | `go·tap` at `+1`; expanded, `tap` at `+1` and `go` at `0`, which stands (R28, R36) | `tap` |
| 1 | `tap` reported as run | `go·tap`, over `go` and the tap | `go·tap²`; expanded, `tap` at `+1` and `go·tap` at `0` | `tap` |
| 2 | `tap` | `go·tap²` | `go·tap³`; expanded, `tap` at `+1` | `tap` |
| 3 | `tap` | `go·tap³` | `go·tap³·done`; expanded, `done` at `+1` and `go·tap³` at `0`: an event, inferred (R40), and no action | nothing |
| 4 | `done`, replacing the inferred one | `go·tap³·done` | nothing: it has no connection | nothing |

Each frame, one voter stands on the apex and holds one connection, and the taps under it are covered and
silent (D10). The match reads each inferred call back: the state it names at `0` stands, its record is
credited, nothing is charged, and the inference is proposed (R49). The loop variable is which of the three
interneurons is standing; the test at the top of the loop is whether that interneuron has ever been followed
by a tap; the exit is the one that was followed by `done`.

A fourth tap, if one ever ran, is paid negatively and lands on whatever connected to it (R32), so the state
that proposed it stops proposing it (R47). With one action in the alphabet the walk (R37) has nothing else to
try, which is right: the answer at frame 3 is not another action but none.

# 5. The stop

**The stop is a thing that follows.** A connection is to what stood on the apex (D25), and nothing stands in
silence: there is no rest value (§3.3). Had the world shown nothing after the third tap, `go·tap³` would hold
no connection at all, and a neuron with nothing to say does not keep the floor: **speaking falls through the
cover**, so the taps under it speak in its place, down to the base (R48). The base `tap` holds the marginal
over every tap it ever fired in, and after a tap a tap followed two times in three, so the fourth tap would
run. The `done` the world shows is what gives `go·tap³` something to infer, and while the coverer speaks,
what it covers is silent. The cue at the end is not optional here; it is how the design learns to stop.

**Which reading of R48.** `go·tap³` infers an event and no action. R48 lets a covered activation speak where
its coverer has no inference for the frame ahead, read per frame: the coverer inferred `done`, so the taps
stay covered, and the action dimension is output nothing (R35). Read per dimension, the coverer would have
nothing for `tap` and the base tap would speak there, and a fourth tap would run. The spec's wording is the
per-frame one, and this case rests on it. A machine that wanted not to rest on it would declare a second
action, `halt`, and have the world run it in the fourth frame: `go·tap³` would then infer an action, the stop
would be chosen by estimate like any action, and both readings agree.

# 6. The same loop, uncounted

Teach the same machine runs of any length, `go`, then taps until the world shows `done` whenever it likes,
and it is a while loop, which is the loop the design runs on its own.

- **Continue.** The chain `go·tap^k` grows an interneuron per frame of the run, with nothing capping it. At a
  length no run has reached before, the interneuron on the apex is new and holds no connection, and speaking
  falls through it to the state below (R48). That state stood uncovered at that age in the runs that went on,
  so it holds the bare tap at `+1`, and the base tap holds a tap after a tap: both say tap. A newly allocated
  interneuron starts with the voices of what it covers, and here the voice says keep going.
- **Stop.** `done` is a neuron that nothing has ever followed with a tap. In the frame it shows, it stands on
  the apex and, with the state beside it, is what the next frame is inferred from; neither infers a tap, and
  nothing runs.

No neuron holds a count, and none is needed: `while (!done) tap()` is one connection at the base and one event
the world controls. The counted loop of §4 is this loop after compression has folded three frames into an
interneuron that knows what came after it.

# 7. The count as data

Show a digit beside `go`, `go 3` or `go 5`, meaning tap that many times, and the machine does it the way
[algorithm-copy.md](algorithm-copy.md) crosses from a digit to a write. `go` and the digit beside it are a
function, `{go, 3@0}`, four numbers for two, and what the first tap sees one back is its interneuron `go·3`.
A class over the digits beside `go` never pays: `m` such functions cost `3m` in the dictionary and two per
occurrence, a class `3 + m` and three per occurrence (R52). So the chain is per digit: `go·3·tap`,
`go·3·tap²`, `go·3·tap³` end in `done`, while `5` has a chain of five, with nothing shared between them but
the base tap. Each count is learned from its own runs, a count never shown has no chain, and nothing in the
machine says that five is more than three. That is the limit the design is honest about: it has no
arithmetic. A count is a length, and a length is learned by seeing it.

# 8. Three in a frame

Inside a frame nothing is back to back. Activations of one frame are at temporal offset zero from one another
and co-occur, with no order among them (D5, D26), and one base action runs per action dimension per position
in a frame (D8). Three taps in one frame are therefore three action dimensions, one channel each (D1): three
tappers, `tap-a`, `tap-b`, `tap-c`, run together. `tap-a` sees the other two at offset zero in every row, so
`{tap-a, tap-b@0, tap-c@0}` is grouped, six numbers for two, and its interneuron `triple` covers them; the
loop of §4 then runs over triples, `go·triple`, `go·triple²`, `go·triple³`, and stops the same way. An order
within the frame that mattered would mean the frame is the wrong tick, and the environment would declare a
finer one.

# 9. The same thing as code

| in the code | in the machine |
|---|---|
| `for (i = 0; i < 3; i++) tap();` | `go·tap`, `go·tap²`, `go·tap³`: a neuron per value of `i` |
| `i` | which interneuron stands on the apex |
| `i < 3` | whether that interneuron has ever been followed by a tap |
| the loop exiting | `go·tap³` connected to `done` and to nothing that runs |
| `while (!done) tap();` | the base tap's connection to itself, and `done`, which no tap has followed |
| `tap(n)` with `n` a number | a chain per `n`, each learned from its own runs; no arithmetic anywhere |
| three statements on one line | three action dimensions at offset zero, covered by one neuron |
