# UCAR on the wavefront specification: implementation and results

What `brain/brain-core` implements of [algorithm.md](algorithm.md) as it stands on the `wavefront` branch at
873967e, the choices made where the text leaves one open, and what the machine does on the six-bar cycle, on
MNIST and on text. The cycle numbers are from `cargo run --release -p brain-core --example simple`; the others
from the Node jobs named with them. The previous implementation, of f8d7f5d, is commit 56de737 on this branch;
its numbers are given beside the new ones.

## What is implemented

The neuron, Parts II and IV, one file per part:

- `pattern.rs`: a function is the owner's activation at offset zero and its other places at rounded offsets
  from it (D15, D16), each a fixed node, a class the table holds, or one place of a parameter of the
  function (D38, D41, D44), written in one canonical order for merging (D43) and for the index (D36). A class
  is a set of members, each with how many uses it stood in. A parameter belongs to its function, ties two or
  more of its places, and keeps as a record the neurons that have stood at all of them within the history.
- `history.rs`: `H` rows, each the activation's coordinate, its own cost, and its neighbors at exact offsets
  with what each cost; a child neighbor also carries the base symbols and neurons it stands for (D22, D47).
  Each row keeps its cover and, per neighbor, the function credited with it and whether it covers or binds
  it (D19, D10).
- `cover.rs`: recognition as the greedy cover (D28) of each row by the table's functions, each measured by
  D22 with nothing besides it taken: the owner where nothing took it, each fixed node covering what stands at
  its offset (or a child standing in for it), each class binding a member at its place, each parameter
  binding the neuron at most of its places, the latest on a tie; every place that holds nothing it could take
  is a failed node at `log₂ |p|`. The price is the bidder, which function, which member per class place and
  once per parameter, and the failed nodes.
- `collapse.rs`: the collapse (D27) decides each place among a fixed node, a class of the table, a new class,
  a parameter tying it to a place already in, and nothing, by what it covers over the uses less a failed node
  in each use it misses and its entry. A class is valued over the uses and over the rows widening to it adds.
  Class members are re-decided over every place that uses the class, with the widening term
  `uses · log₂(|K'| / |K|)`. A function with a child never names a neuron built over it. Re-centering runs
  the collapse on every function whose rows changed (D29).
- `margin.rs`: margin as what the cover saves with the function less the greedy cover without it, over the
  rows it covers, less its entry (D30); every negative margin retired together; a class no function uses, or
  left with one member, deleted (R18, R41).
- `merge.rs`: identical classes and identical functions merged into the older (D43).
- `pick.rs`: the relation index (D47) over the residual, of three kinds (node to node, the owner at one end
  or not; same to same; node to open, from the two neurons that stood at a place most), a child counted as
  itself and as each base symbol it stands for at an equal share of its cost; ranked by D22's savings of the
  relation as a whole function (D33), then count, older neuron, shorter offset; each tried in that order and
  grown by the collapse; kept where the cover with it saves more than as it stands by more than its entry
  and the entries of the classes it made (R14, R15).
- `bid.rs`: a bid for every function that pays against the activation's neighborhood with nothing taken,
  in the cover or not (D31, R21).

The machine, Parts III and V:

- `election.rs`: the election is D28 over the board (R22, R24): the coverage set credits each activation once,
  as covered or as bound (D24). The child fires at its bidder's coordinate (D53), costing the bidder, which
  function and its failed nodes; its arguments are what stood in its variables, which stay on the apex. A
  child is reused through the index before one is minted (R16, R43). Span per neuron from its functions (D4).
- `expansion.rs`: the owner at the child's coordinate, each other place at its offset (R28); a fixed node
  expands to its neuron, a variable to its argument; a call with no argument given reads the arguments off
  its activation where one stands there, and is otherwise bound as an inferred call: a parameter from what
  stands at those of its places that have happened, a class to its most frequent member.
- `index.rs`: keys with the variables left open (D36), recomputed after every frame, an interneuron that comes
  to stand for what an older one does rewired to it.
- `neighborhood.rs`, `connect.rs`, `selection.rs`, `death.rs`: as before, over the new functions; the match
  (R49) reads fixed nodes, classes and parameters as the table in R49 says.
- The dispatch is unchanged: regions of columns, each level's calls fanned out over the columns with Rayon,
  the election, wiring and ledger sequential.

## Choices made where the text leaves one open

- **Failed nodes everywhere.** A function holds in a row with some places failed, each at `log₂ |p|`, in
  recognition, in bids and on the board. A function bids where it pays against the whole neighborhood
  with nothing taken.
- **Uses of the collapse are the rows where every place holds** (D27's words). Without this, re-centering
  over every row a function covers, failed ones included, collapsed every new function to nothing within a
  frame.
- **Widening.** A class place is valued over the function's uses and over every other row where the rest
  of the function holds and a candidate stands unowned at the place (D27: "the function then holds in rows
  it did not").
- **A weak executed pattern at the owner's coordinate is not its neighbor.** R30 fires it at the frame's
  level 0, where the owner's own row sees it at offset zero standing for the owner itself; tables then named
  their own call at offset zero. An interneuron whose expansion places the owner's activation is left out of
  the owner's neighborhood.

## The six-bar cycle

One channel, a binary price move and own/out, the `eventsToActions` policy, the world running the right
action where the machine is silent, the reward the bar's return signed by the action. Over 150 laps,
right answers in the last ten laps, of 60:

| `H` \ radius | 2 | 3 | 6 | 12 |
|---|---|---|---|---|
| 64 | 30 | 39 | 43 | 45 |
| 256 | 30 | 47 | 53 | 57 |
| 1024 | 30 | 36 | 50 | 56 |

Over 500 laps at `H` 256: radius 12, 83.1% overall and 54 of 60 in the last ten laps; radius 6, 75.1% and 45
of 60. Nothing settles: at radius 12 the run added 2,643 functions and retired 2,546, about five of each per
lap, and minted 1,379 neurons to keep 11 alive. The previous implementation froze with one bar wrong (50 of
60) at every radius from 3 to 12.

**Why it churns.** Two things, both read from the machine's trace:

1. *The collapse and the margin price a function over different rows.* The collapse decides a place over the
   uses, the rows where every place holds, and adds it where it pays there. The margin is taken over every row
   the function covers, failed ones included. A place added for the uses raises `|p|` and is one more failed
   node in every other covered row, so the next margin is below the entry and the function is retired: for
   instance `F[me #0@-4 #1@-3]`, built at margin +5.5, re-centered to `F[me #0@-4 #1@-3 #307@-1]` over its 12
   uses, then benefit 9.5 against entry 19.5. Each rebuild mints its child anew.
2. *A class of the whole alphabet pays at every offset.* A neighbor costs `log₂` of all neurons; stating
   which member of a class costs `log₂ |K|`. With two event neurons a class of both costs one bit and covers
   two, so the collapse turns most places into "either symbol here", and the call neuron of such a function
   says nothing about the phase.

At radius 2, nothing forms at all: eight functions in 150 laps, every bar's answer from the base
connections alone, 30 of 60.

## MNIST

**Ten images, one per digit, 7×7, 500 episodes** (`apps/mnist/jobs/mnist.js --image-size 7 --per-class 1
--teach 10 --test 10 --episodes 500`, `H` 2048): 7 of 10 from episode 50 to 500, the 9, 2 and 8 wrong; from
episode 50 on nothing changes (32 neurons, 43 functions, 110 connections). The three wrong images' answers
walk among digits with estimates near zero. The previous implementation had 8 of 10.

**Why the 9 stays wrong** (the machine's trace, episode 55). At the 9 the electorate is one child of height 2,
`#27`, and four ink pixels the bids bound, which stay on the apex and vote (D10, R46). The four pixels are the
one ink neuron, whose connections are one set over every image: digit 9 at −0.56. `#27` has connections to
nine digits and none to 9; three of them were wired by the walk and never ran, so they sit at 0.00. Resolution
takes the share-weighted estimate per digit (R47): digit 9 has only the ink neuron's −0.56, and the digits
`#27` names at 0.00 beat it, so 9 never runs, never connects, and the walk, which wires a new action only when
an estimate turns negative (R37), never reaches it, since the zero estimates never move.

The tables are nearly all parameters. An image has one ink neuron, so a parameter over a set of places is
"ink at all of these", the same as fixed nodes there, but its entry is an offset per place where a fixed node
costs a neuron besides, and its record holds one member, so its choice is free (D44: "a parameter costs
nothing beyond its relations").

**10,000 images, 7×7, one episode** (`--per-class 1000 --teach 200 --test 500`, `H` 2048): 17.8% of the 9,642 training images it answered, falling from about 20% over the first 1,200 to
17.8% by the end; on test 13.3% of the 467 it answered (62 of 500). Chance is 10%. The previous
implementation had 11.6% on the same test. Training took 18 minutes, the time per image rising from 12 ms
to about 150 ms; the run ended with 81 neurons and 118 functions.

## Text

`apps/text/jobs/text.js` on `test.txt` (127 bytes), `H` 256, radius 1, next byte right over twelve passes:
17.9, 67.5, 72.2, 66.7, 57.9, 56.3, 52.4, 53.2, 54.8, 50.8, 50.8 and 47.6%. It peaks at the third pass and
falls while the apex keeps shrinking (5.2 bits a frame on the first pass, 1.8 on the ninth). The previous
implementation had 21, 61, 68, 69 and 80% over five.

The letter stream `th ch sh wh` four times (`examples/text.rs`): 31 of 47 (previously 29). `h`'s table
holds a class at −1 of the letters before it where none of them recurred alone first; where `t` recurs before
the others have stood, `t` gets its own function first and the others' rows are owned before a class can
seed, so which forms depends on arrival order (`varied_predecessors_form_a_class` passes or fails with the
pick's order).

`abramov_summary.txt` (2 KB), two passes: 17.3% and 20.7% (previously 17.7% over two), 3.3 and 6.5 seconds.

## Tests

`cargo test --release -p brain-core`: three of five pass. `six_bar_cycle_settles` fails (53 of 60 in the
last ten laps), and `varied_predecessors_form_a_class` fails at the end of the stream, where the class at −1
has no child and the predecessors are named by their own functions.

## Not implemented, or simplified

- Keys for the index and spans are recomputed for every living interneuron after each frame.
- The owner's reach in prices (D13) is read from its span as its functions stand.
- A parameter's record is refreshed from the whole history at each call; a change in it does not re-cover
  rows already covered.
- No backup or persistence.

## Running it

```bash
cd brain
cargo test --release -p brain-core
cargo run --release -p brain-core --example simple -- 150 256 6
cargo run --release -p brain-core --example text -- "th ch sh wh th ch sh wh th ch sh wh th ch sh wh " 1 64 h
```
