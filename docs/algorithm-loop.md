# Tap three times and stop, as a machine that could be built

A worked case for [algorithm.md](algorithm.md): a machine that, shown a cue, runs one action in each of the
next three frames and then runs nothing, written out neuron by neuron. Nothing here is normative, and the
machine is shown running: its tables and connections are as listed, and its histories already hold runs like
this one. It shows what a loop is in this design. There is no counter: the count is a chunk, the loop variable
is which chunk stands on the apex, and the loop ends because the last chunk has never been followed by the
action. The same machine, taught to stop on a cue instead of a count, is a while loop, and that is the one the
design runs natively.

---

# 1. The environment

**In.** One channel, laid out over time alone, with radius `R = 1` (D1): a neuron sees the frame before it.
It declares one event dimension, `cue`, with two buckets, `go` and `done`.

**Out.** One action dimension, `tap`, with one base action, `tap`, in the same channel, so a tap is a neighbor
of the cue and of the tap before it (D1). One base action runs per action dimension per position in a frame (D8), and `tap` has time alone, so a tap is
one per frame: what the loop counts is frames.

**Teaching.** The world shows `go`, runs `tap` itself in each of the next three frames and rewards each
(§3.5), shows `done` in the fourth, and shows nothing for a while. Then again. A dozen runs are enough for every
line below to pay.

The channel is one of several the machine holds, so a base symbol is priced among an alphabet of some size,
say thirty-two, 5 bits (D13). Alone, with three symbols, nothing here would pay for its own line.

# 2. The neurons

| Kind | Neurons |
|---|---|
| base events | `go`, `done` |
| base action | `tap` |
| the count | three functions in `tap`'s table, each naming one neighbor a frame back: `go`; `go·tap`; `go·tap²`. Their children `go·tap`, `go·tap²` and `go·tap³`, at levels 1, 2 and 3 |
| the end | a function in `done`'s table naming `go·tap³` a frame back; its child `go·tap³·done` |

That is the whole machine: three neurons for the three states of the loop, and one for its end.

# 3. How it was learned

1. **The first tap names the cue.** In `tap`'s history a third of the rows hold `go` one back, and a row of
   one neighbor carries the relation of the owner with that neighbor (D47). Its collapse is a function naming
   `go` at `−1` (D27); it pays at its second use against a line of one neuron and one offset (R15); its bid
   covers `go` and the tap, and the child `go·tap` fires at the tap's frame, one level up (R16, §7.4).
2. **The second tap names the first chunk.** Once `go·tap` fires, the next tap sees it one back, since a
   neighbor is whatever stands uncovered within reach, at whatever level (D5). Another third of `tap`'s rows
   come to hold `go·tap` at `−1`, and the same relation gives the function and the child `go·tap²`, covering
   `go·tap` and the tap: two levels up, three frames long (D2).
3. **The third tap names the second.** `go·tap³`, the same way, a level higher. Each chunk is one letter
   longer than the one below and stands one level above it, as a word does in
   [algorithm-text.md](algorithm-text.md). The chain grows a level every couple of runs, because a chunk has to
   exist and fire before the tap after it can see it.
4. **The end names the last chunk.** `done` always sees `go·tap³` one back, and its table holds the function;
   the child `go·tap³·done` covers both. Nothing ever follows it within reach but silence, so it holds no
   connection, and needs none.
5. **Every state learns what came next.** Each chunk was open and uncovered in the frame after it fired, and in
   that frame something covered it: `go` by `go·tap`, `go·tap` by `go·tap²`, `go·tap²` by `go·tap³`, and
   `go·tap³` by `go·tap³·done`. A neuron covered in a frame connects once, to the neuron that covered it, with
   that frame's reward (D10). So each state holds exactly one connection at offset one, to the state after it,
   and the first three carry the tap's reward.

Nothing in `tap`'s table says three. `tap` holds three functions because three different things have stood
before a tap, and the chunks they build are as long as the runs the world showed.

# 4. One run, frame by frame, after teaching

| Frame | input | on the apex | infers for the frame ahead | output |
|---|---|---|---|---|
| 0 | `go` | `go` | `go·tap` at `+1`; expanded, `tap` at `+1` and `go` at `0`, which stands (R28, R36) | `tap` |
| 1 | `tap` reported as run | `go·tap`, over `go` and the tap | `go·tap²`; expanded, `tap` at `+1` and `go·tap` at `0` | `tap` |
| 2 | `tap` | `go·tap²` | `go·tap³`; expanded, `tap` at `+1` | `tap` |
| 3 | `tap` | `go·tap³` | `go·tap³·done`; expanded, `done` at `+1` and `go·tap³` at `0`: an event, expected (R40), and no action | nothing |
| 4 | `done`, replacing the expectation | `go·tap³·done` | nothing: it has no connection | nothing |

Each frame, one voter stands on the apex and holds one connection, and the base tap under it is covered and
silent (D10). The loop variable is which of the three chunks is standing; the test at the top of the loop is
whether that chunk has ever been followed by a tap; the exit is the chunk that was followed by `done`.

A fourth tap, if one ever ran, is paid negatively and lands on whatever connected to it (R32), so the state that
proposed it stops proposing it (R47). With one action in the alphabet the walk (R37) has nothing else to try,
which is right: the answer at frame 3 is not another action but none.

# 5. The stop

**The stop is a thing that follows, not an absence.** A connection is to what stood on the apex (D25), and
nothing stands in silence: there is no rest value (D10). Had the world shown nothing after the third tap,
`go·tap³` would hold no connection at all, and a neuron with nothing to say does not keep the floor:
**speaking falls through the cover**, so the taps under it speak in its place, down to the base (R48). The
base `tap` holds the marginal over every tap it ever fired in, and after a tap a tap followed two times in three,
so the fourth tap would run. The `done` the world shows is what gives `go·tap³` something to infer, and while
the coverer speaks, what it covers is silent. The user's event cue is not optional here; it is how the design
learns to stop.

**Which reading of R48.** `go·tap³` infers an event and no action. R48 lets a covered activation speak where
its coverer "infers nothing for `f + 1`", read per frame: the coverer inferred `done`, so the taps stay covered,
and the action dimension is output nothing (R35). Read per dimension, the coverer would have nothing for `tap`
and the base tap would speak there, and a fourth tap would run. The spec's wording is the per-frame one, and
this case rests on it. A machine that wanted not to rest on it would declare a second action, `halt`, and have
the world run it in the fourth frame: `go·tap³` would then infer an action, the stop would be chosen by
estimate like any step, and both readings agree.

# 6. The same loop, uncounted

Teach the same machine runs of any length, `go`, then taps until the world shows `done` whenever it likes, and
it is a while loop, which is the loop the design runs on its own.

- **Continue.** The chain `go·tap^k` grows a neuron per frame of the run, with nothing capping it (T13 in
  [algorithm-remarks.md](algorithm-remarks.md); "depth on input that repeats" in
  [algorithm-evaluation.md](algorithm-evaluation.md)). At a length no run has reached before, the chunk on the
  apex is new and holds no connection, and speaking falls through it to the first chunk below that has one, or
  to the base tap, whose marginal says tap (R48). A newly minted child starts with its parents' voices, and
  here the voice says keep going.
- **Stop.** `done` is a neuron that nothing has ever followed with a tap. In the frame it shows, it stands on
  the apex and, with the chunk beside it, is what the next frame is inferred from; neither infers a tap, and
  nothing runs.

No neuron holds a count, and none is needed: `while (!done) tap()` is one connection at the base and one event
the world controls. The counted loop of §4 is this loop after compression has folded three frames into a chunk
that knows what came after it.

# 7. The count as data

Show a digit beside `go`, `go 3` or `go 5`, meaning tap that many times, and the machine does it the way
[algorithm-copy.md](algorithm-copy.md) crosses from a digit to a write. `go`'s table holds a class of the digits
beside it (D41), and its value neurons `go:3`, `go:5` are what the first tap sees one back. The chain is then
per value, `go:3·tap`, `go:3·tap²`, `go:3·tap³`, ending in `done`, and `go:5` has a chain of five, with nothing
shared between them but the base tap. Each count is learned from its own runs, a count never shown has no chain,
and nothing in the machine says that five is more than three. That is the limit the design is honest about: it
has no arithmetic. A count is a length, and a length is learned by seeing it.

# 8. Three in a frame

Inside a frame nothing is back to back. Activations of one frame are at temporal offset zero from one another
and co-occur, with no order among them (D5, D26), and one base action runs per action dimension per position in a frame (D8). Three
taps in one frame are therefore three action dimensions, one channel each (D1): three tappers, `tap-a`,
`tap-b`, `tap-c`, run together. In `tap-a`'s table the other two stand at offset zero in every row, so a function
names them and its child, `triple`, covers the three (§7.1); the loop of §4 then runs over triples, `go·triple`,
`go·triple²`, `go·triple³`, and stops the same way. An order within the frame that mattered would mean the
frame is the wrong tick, and the environment would declare a finer one.

# 9. The same thing as code

| in the code | in the machine |
|---|---|
| `for (i = 0; i < 3; i++) tap();` | `go·tap`, `go·tap²`, `go·tap³`: a neuron per value of `i` |
| `i` | which chunk stands on the apex |
| `i < 3` | whether that chunk has ever been followed by a tap |
| the loop exiting | `go·tap³` connected to `done` and to nothing that runs |
| `while (!done) tap();` | the base tap's connection to itself, and `done`, which no tap has followed |
| `tap(n)` with `n` a number | a chain per `n`, each learned from its own runs; no arithmetic anywhere |
| three statements on one line | three action dimensions at offset zero, chunked into one neuron |
