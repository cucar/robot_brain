# UCAR on the wavefront specification: implementation and results

What `brain/brain-core` implements of [algorithm.md](algorithm.md) as it stands on the `wavefront` branch at
f8d7f5d, with no departures, what is not there yet, and what the machine does on the six-bar cycle, on MNIST
and on text. The cycle numbers are from `cargo run --release -p brain-core --example simple`; the others from
the Node jobs named with them.

## What is implemented

The neuron (`neuron.rs`), Parts II and IV:

- A pattern as a set of relations over nodes (D15, D16, D34): a function is its nodes with their offsets
  from a reference node, the owner's activation where it names it and otherwise the node latest in time,
  written in one canonical order for merging (D43) and for the index (D36). A class is its members (D41). A
  parameter is its offsets from its first position, with its members as a record (D44).
- The history (D18): `H` rows, each the activation's coordinate, its own cost, and its neighbors at exact
  offsets with what each cost, its cover (D17), owners (D19) and the coordinates its variables claim.
- Recognition as the greedy cover (D28) over each row: every function at every place one of its nodes
  holds, the owner anchoring it where named (§5.2), with its variables, each node holding or failing and
  each failure priced at `log₂ |p|`; then every parameter alone on what no function took (D31). A class is
  never a claimant alone. Measured by D22, with the coarse-offset groups of D26. Where a pattern names a
  neuron, a child standing for that neuron at that offset matches as well, and is credited once (D22).
- Uses as places (D27): the coordinates a pattern's nodes stand at, counted once however many rows saw
  them, the latest row credited.
- The collapse per kind (D27): a function decides every candidate relation on its own over the seed's
  uses, taken where what it covers beats a failed neighbor in each use it misses and its entry; the uses
  that count are those where no other function names the coordinate, and a candidate being created skips
  every owned neighbor. A parameter decides its positions, a class its members with the widening term. A
  line with a child never names a neuron built over that child, and a variable never takes one built over
  its own value neurons.
- The three seed relations (D47), a child on the apex a node as itself and as each base symbol it stands
  for within the owner's reach, its cost split equally; the owner's activation a node where nothing was
  credited it. The two-stage pick with variables first and the claims they leave (D33), seeds ranked by
  D22's savings as they stand summed over their uses, every seed with two uses tried; candidates priced by
  the exact difference between a use's cover with the candidate in it and as it stands (R14, R15).
- Margin (D30) as the difference with and without the pattern over its uses, every pattern with a
  negative margin retired together (R18, R41), merging (D43, §5.5), and the return: one bid per function
  with its variables, functions that share a variable together as one bid, and one per parameter alone
  (D31, R21).

The machine (`machine.rs`), Parts III and V:

- Base neurons by declaration (D52), the interneuron index by what each stands for (D36), with keys
  recomputed after each frame's levels and lines whose child comes to stand for what another neuron does
  rewired to it.
- Neighborhoods (D5, D7): every activation in the same frame or an earlier one, within reach in every
  shared dimension, both on the apex, the past table of the policy applied (D50).
- Levels (R26): activate, call, elect, create, until a level fires nothing, the neurons' calls of one level
  dispatched over columns in parallel. The election is D28 over the board with R22's measure, failed nodes
  priced, and R24's tie-break; credit once, never revisited (R23).
- Children placed at the latest frame covered and the center of the box in space (D53), with their span
  read from what their nodes stood for (D4); value neurons where the member stood; reuse through the index
  before anything is minted (R43, R16); the death ledger and the subtree rule, the ledger holding a retired
  line readable until every window over what it fired has closed (R38).
- Connections written by the connect call to the apex of each frame within reach, the last connection to
  the coverer (D10, R31), rewards shaped by span and scoped by channel (R33), the walk when an estimate
  turns negative (R37); a child placed frames back is handed the frames it missed (§7.1).
- Inference at age 0 (R36), standing inferences (R45) and the match that strikes them (R49), expansion
  down to base symbols (R28), the electorate falling through covers (R46, R48), resolution by estimate and
  by share (R47), weak activations that last their frame (D40, R40), and a pattern an inference's expansion
  completes at the frame ahead firing weakly there where what it placed was selected (R30).

## What changed from the previous version

The previous version (16f8f22) departed from the spec in four ways and carried three fixes the spec
did not ask for. This one removes them: holding tolerates failed nodes at `log₂ |p|` each (D22); the
collapse decides each relation on its own and never narrows a line's uses (D27), and nothing is done to a
narrowed line's child; a class is never bid alone; retirement takes every negative margin together (R18);
a variable no function names is not dropped at the end of the pick but retired by its margin; seeds are
ranked by D22's savings over every use. Two departures are in the spec now and were widened to its words:
a child stands in for any node of its expansion (D22, D47), and every apex activation of a frame is
adjacent to the others (D5). New in the spec and implemented: no line names a neuron built over its own
child (D27, D29), no once-per-descent rule in expansion, the index or the span (R28, D4). Never
implemented before and implemented now: the match (R49), shared-variable bids (D31), weak firing of a
completed pattern (R30), the ledger holding retired lines (R38), and the walk only when an estimate turns
negative (R37).

Five faults of the implementation were found and fixed while testing this version:

- **The seed estimate counted the owner twice.** A function seed naming the owner counted the owner's cost
  as covered and left the bidder's cost off the price, so every owner seed looked one activation better
  than D22 says.
- **Stand-ins reached past the box.** A child's base symbols were neighbors at any offset, so a base neuron
  of reach 6 named offsets of 7 and beyond, which D15 forbids (a pattern fits inside the box its
  neighborhood spans).
- **A stand-in carried its child's span.** Where a child stood in for a base symbol the line named, the
  new child's span took the stand-in's whole span, so on a repeating input each lap added a lap: spans
  reached 894 frames at lap 150 and reach with them. D4 reads the span of what the line names, a base
  symbol's being zero.
- **D27 read a snapshot.** Whether a neighbor is built over a line's child was saved in the row when the row
  was recorded, so a neighbor that came to be built over the child later passed the check, and a circular
  definition formed. It is now read from the index as it stands when each level begins.
- **Index keys expanded classes.** A class was written into a key through each member's own key, so a
  function naming a class whose member is built over the function's child gave a key that contains itself.
  A call stands for its relations with the variables left open (D36); a class is now named by its members.

## Not implemented, or simplified

- Lines re-center in parallel within a level, each against the index as the level began, so two lines
  re-centering in one level can still close a circular definition between them; expansion stops where it
  meets a neuron again and counts it. On text it happened 11 times in 4,175 frames.
- Keys for the index are recomputed for every living interneuron after each frame; nothing is incremental.
- The owner's reach in prices (D13) is read from the largest span any activation of the neuron has had.
- No backup or persistence. The dispatch is parallel: the thalamus owns regions, a region its columns, a
  column its partition of the neurons by id, and the call of a level fans out over the columns with Rayon;
  the election, wiring and ledger run sequentially on what comes back.

## The six-bar cycle

One channel, a binary price move and own/out, the `eventsToActions` policy, the world running the right
action where the machine is silent, the reward the bar's return signed by the action, scoped to the channel
and the frame. At `H` 256 and radius 6, over 150 laps:

| laps | right per bar, of 10 | added | retired |
|------|----------------------|-------|---------|
| 1-10 | 3 0 8 8 9 10 | 6 | 0 |
| 11-20 | 10 0 10 10 10 10 | 0 | 0 |
| 21-150 | 10 0 10 10 10 10 | 0 to 10 | 0 to 11 |

The bar decided at bar 1 is wrong in every lap; the other five are right from lap 11. Over 300 laps:

| `H` \ radius | 2 | 3 | 4 | 6 | 8 | 12 |
|---|---|---|---|---|---|---|
| 64 | frozen, one bar wrong | frozen, one bar wrong | frozen, one bar wrong | frozen, one bar wrong | frozen, one bar wrong | frozen, one bar wrong |
| 256 | all right from lap 201 | frozen, one bar wrong | frozen, one bar wrong | frozen, one bar wrong | frozen, one bar wrong | frozen, one bar wrong |
| 1024 | all right from lap 263 | frozen, one bar wrong | frozen, one bar wrong | frozen, one bar wrong | frozen, one bar wrong | frozen, one bar wrong |

The previous version, with its departures, was right from lap 11 at radius 6 for every `H` of 64 or more.

**Why bar 1 stays wrong.** The cycle is up, down, down, up, up, down. The up neuron fires at bars 1, 4
and 5. Its first line names the base symbols at −1, −3, −4 and −6 from it, and those are the same before
bar 1 and before bar 4; the two differ at −2 and −5. So one line, and one child, stands for both bars, and
the child's connection to "up" averages a gain after bar 4 with a loss after bar 1, +0.51. Four things keep
it there, each as the specification says:

1. On the apex at bar 1 stands one activation within reach, the child of the frame before; everything
   earlier was covered by the frame after it. The line's base symbols are all matched through that child
   standing in for them (D22), so the bid covers it, and nothing in the row is left residual.
2. With nothing residual, nothing seeds (D47): no second line can form in the up neuron's table.
3. Re-centering cannot add what tells the bars apart. The children at −1, −2 and −5 are each built over
   the line's own child, and a line never names a neuron built over its own child (D27). The base symbols
   at −2 and −5 reach the line only as a child's stand-ins, at a share of its cost under a bit, which never
   pays for the failed node it costs in the other bar's uses.
4. The child's estimate for "up" is positive, so it never turns negative and the walk never wires "down"
   (R37).

At radius 2 the line that fits both bars cannot form, since −3 and beyond are out of reach, and the bars are
told apart by lines over the two frames before.

## MNIST

**Ten images, one per digit, 7×7, 500 episodes** (`H` 2048, radius 1, the digit taught on the first
episode only): 6 of 10 at episode 25, 8 of 10 from episode 100 to 500, the 5 answered as 6 and the 8 as 2
throughout, about 30 lines added and retired every episode. The previous version had all ten from episode 25.

**10,000 images, 7×7, one episode** (`H` 2048, the first 200 taught), 500 test images: per 1,000 training
images 14.2, 11.8, 12.1, 12.2, 13.7, 15.6, 11.6, 12.3, 14.3 and 12.1% of those it answered; on test 11.6% of
the 467 it answered. Chance is 10%. The previous version had 20.8% on the same test.

## Text

`apps/text/jobs/text.js` on `apps/text/data/test.txt` (127 bytes), `H` 256, radius 1: 21.1, 61.1, 67.5,
69.0 and 80.2% of next bytes right over five passes; the previous version had 20 and 78% on its first two.
The letter stream `th ch sh wh` four times: 29 of 47.

`abramov_summary.txt` (2 KB, its ASCII characters) streamed twice through the text example: 737 of 4,175
right, 17.7%, the second pass no better than the first, which alone was 18.6% through the Node job. Late
frames take up to 12 seconds: the pick tries every seed with two uses (D33), and an interneuron with a
long history holds several hundred, of which almost none builds.

## Running it

```bash
cd brain
cargo test --release -p brain-core                       # five behavioral tests, the cycle among them
cargo run --release -p brain-core --example simple -- 150 256 6
UCAR_TRACE=1 cargo run --release -p brain-core --example simple -- 40 256 6               # adds, retires with reasons, deaths, cascades
UCAR_DUMP=1  cargo run --release -p brain-core --example simple -- 150 256 6              # every table at the end
cargo run --release -p brain-core --example text -- "th ch sh wh th ch sh wh th ch sh wh th ch sh wh " 1 64 h
```
