# UCAR on the wavefront specification: implementation and results

What `brain/brain-core` implements of [algorithm.md](algorithm.md) as it stands on the `wavefront` branch,
the one departure it makes, what is not there yet, and what the machine does on the six-bar cycle and on a
letter stream. The numbers below are from `cargo run --release -p brain-core --example simple` and
`--example text`; the Node jobs give the same.

## What is implemented

The neuron (`neuron.rs`), Parts II and IV:

- A pattern as a set of relations over nodes (D15, D16, D34): a function is its nodes with their offsets
  from a reference node, the owner's activation where it names it and otherwise the node latest in time,
  written in one canonical order for merging (D43) and for the index (D36). A class is its members (D41). A
  parameter is its offsets from its first position, with its members as a record (D44).
- The history (D18): `H` rows, each the activation's coordinate, its own cost, and its neighbors at exact
  offsets with what each cost, its cover (D17), owners (D19) and the coordinates its variables claim.
- Recognition as the greedy cover (D28) over each row: every function at every place its relations are
  satisfied, the owner anchoring it where named (§5.2), with its variables; then every parameter alone on
  what no function took (D31). Measured by D22, with the coarse-offset groups of D26.
- Uses as places (D27): the coordinates a pattern's nodes stand at, counted once however many rows saw
  them, the latest row credited.
- The collapse per kind (D27): a function re-decides every candidate relation from a node already in to
  what stands around its uses, the owner's activation among the candidates at each offset rows saw the
  use from; a parameter its positions; a class its members with the widening term.
- The three seed relations (D47), the two-stage pick with variables first and the claims they leave
  (D33), candidates priced by the exact difference between a use's cover with the candidate in it and as
  it stands (R14, R15), seeds that build nothing passed over.
- Margin (D30) as the difference with and without the pattern over its uses, retirement (R18, R41),
  merging (D43, §5.5), and the return: one bid per claimant of each of the frame's covers (D31, R21).

The machine (`machine.rs`), Parts III and V:

- Base neurons by declaration (D52), the interneuron index by what each stands for (D36), with keys
  recomputed after each frame's waves and lines whose child comes to stand for what another neuron does
  rewired to it.
- Neighborhoods from the open activations within the activation's own reach (D4, D5, D7), the past table
  of the policy applied (D50).
- Waves (R26): activate, call, elect, create, until a wave fires nothing. The election is D28 over the
  board with R22's measure and R24's tie-break; credit once, never revisited (R23).
- Children placed at the latest frame covered and the center of the box in space (D53), with their span
  read from what they cover (D4); value neurons where the member stood; reuse through the index before
  anything is minted (R43, R16); the death ledger and the subtree rule (R38).
- Connections written by the connect call to the apex of each frame within reach, the last connection to
  the coverer (D10, R31), rewards shaped by span and scoped by channel (R33), the walk on a negative
  estimate (R37); a child placed frames back is handed the frames it missed (§7.1).
- Inference at age 0 (R36), standing inferences (R45), expansion once per descent with the owner's symbol
  placed where a chain closes on itself (R28), the electorate falling through covers (R46, R48),
  resolution by estimate and by share (R47), and weak activations that last their frame (D40, R40).

## The one departure

D18 records the apex at firing; D5 makes an activation another bid already covers no neighbor at all.
Here a row records every activation within reach, with a flag saying whether it stood on the apex when the
owner fired. Holding reads every recorded activation. Seeding (D47) and credit (D20, D22) read only the
uncovered ones. On the board R22 already says the same thing: a neuron the bid names that fired and
another bid covers is credited to no one and charged nothing. The change makes the neuron's arithmetic
match the machine's, and nothing else.

One consequence showed up at once in the traces and is worth stating. The collapse decides what a line
names by what each node covers (D27). A node whose activations are always covered by other bids covers
nothing, so the collapse drops it at the next re-centering even though it holds in every use. A line
whose nodes are all out-credited therefore degenerates to the owner alone and is retired, which is the
same end the document's rule reaches by failure, one step later.

## Not implemented, or simplified

- The match (R49): standing inferences are not read back against the frames they point into and are
  never struck; they stand until the window closes.
- Functions of one cover that share a variable go as separate bids, not as one (D31).
- Keys for the index are recomputed for every living interneuron after each frame; the window scan of the
  earlier implementation is gone, but nothing is incremental yet.
- Spans are per activation (D4), and the owner's reach in prices (D13) is read from the largest span any
  activation of the neuron has had.
- No backup or persistence, and no parallel dispatch: neurons are called one at a time per wave.
- The MNIST job runs but has not been measured on this core.

## The six-bar cycle

The configuration of the specification: one channel, a binary price move and own/out, radius 6, `H` 256,
the `eventsToActions` policy, the world running the right action where the machine is silent, the reward
the bar's return signed by the action, scoped to the channel and the frame. 200 laps:

| laps | right per bar, of 20 | minted | added | retired | alive neurons | patterns | max height |
|------|----------------------|--------|-------|---------|---------------|----------|------------|
| 1-20 | 13 15 17 17 18 16 | 20 | 24 | 12 | 14 | 9 | 5 |
| 21-40 | 17 16 20 19 20 15 | 20 | 58 | 40 | 14 | 13 | 4 |
| 41-60 | 11 17 20 18 20 17 | 18 | 134 | 118 | 24 | 23 | 5 |
| 61-80 | 16 18 20 20 20 14 | 21 | 170 | 160 | 31 | 26 | 6 |
| 81-100 | 16 20 19 18 20 17 | 12 | 165 | 155 | 31 | 30 | 8 |
| 101-120 | 14 18 20 20 20 16 | 14 | 44 | 32 | 42 | 38 | 10 |
| 121-140 | 14 18 20 20 20 19 | 12 | 102 | 96 | 41 | 41 | 11 |
| 141-160 | 15 15 20 18 20 16 | 11 | 182 | 169 | 39 | 39 | 11 |
| 161-180 | 14 20 20 19 20 18 | 17 | 234 | 226 | 43 | 42 | 7 |
| 181-200 | 13 19 19 18 20 18 | 14 | 312 | 297 | 35 | 32 | 3 |

Overall 1065 of 1197 right (89%), the last ten laps 55 of 60. Bars 3 and 5 are right from lap 10 on; bars
1 and 6 keep being lost, since the lines that serve them are the ones that turn over. The machine does not
settle: patterns are added and retired at a rising rate, and the stack of children over the base grows to
height 11 and collapses again.

## Four readings of the neighborhood, same outcome

Three switches select how a row is recorded and read; all four combinations were run on the cycle for
100 laps (`UCAR_ROWS=apex`, `UCAR_THROUGH=1`, `UCAR_OWNER_PARAM=1` on the example):

| rows | parent matches child | owner as a parameter position | right, laps 81-100 (of 120) | added / retired over 100 laps | alive neurons at 100 |
|------|----------------------|-------------------------------|-----------------------------|-------------------------------|----------------------|
| apex only, as D18 says | no | no | 103 | 1683 / 1597 | 37 |
| every activation, flagged | no | no | 110 | 551 / 485 | 31 |
| apex only | yes | no | 102 | 1057 / 977 | 38 |
| every activation, flagged | yes | no | 110 | 572 / 507 | 34 |
| every activation, flagged | no | yes | 110 | 323 / 255 | 51 |
| apex only | no | yes | 111 | 3516 / 3430 | 30 |
| every activation, flagged | no | no, and a function naming the owner may not name the owner's own neuron | 109 | 153 / 90 | 25 |
| apex only | no | same rule | 108 | 1762 / 1678 | 43 |

The last rule (`UCAR_NO_SELF=1`) excludes exactly the lines that build the stack, "owner, and the owner's
own neuron one period back". With it and flagged rows the base tables hold, by lap 40, one line per bar
over base symbols, "owner down, up at −2, up at −1" for bar 6 and "owner up, down at −4, down at −2,
down at −1" for bar 4 among them, which is the structure the discussion asked for; the children's
tables then build lines over other children, and the stack continues a level every ten laps or so.

The document's own reading churns the most. Letting a pattern's parent be matched by the child covering
it does not change the race described below, since the child that covers the parent is the one the base
line then names, and the child's own table runs first either way. Admitting the owner's activation as a
parameter position turns the self-period relation into a parameter whose value neuron the index reuses,
which lowers the churn and raises the neuron count, and the tables still turn over.

## What the traces show

The per-lap dump (`UCAR_LAP=1`) of the two base tables and the apex per bar, read with the add and retire
trace (`UCAR_TRACE=1`), gives the sequence. Up is neuron 1, down is 0; the actions are 2 and 3 and under
this policy are never covered.

1. **Lap 2: the two bigrams.** Down's table adds "owner, up at −1", which holds at bars 2 and 6; up's table
   adds "owner, down at −1", which holds at bars 1 and 4. Each is bought a child, 4 and 5. From here every
   base activation is covered by its own bar's bid, and the next bar sees the child.
2. **Lap 4: the child names itself one period back.** Child 4 fires at bars 2 and 6, so from lap 3 its own
   activation stands uncovered at −6 in every row. That is a relation counted twice per lap (D47), it pays
   after three uses, and 4's table adds "owner, 4 at −6" with a child, 6, in the same wave the child fires.
   The base lines that would name 4 at −1 need the same three uses at one or two per lap, and lose the race
   by a frame. Child 5 does the same a lap later. Over the run, 28 of the 107 functions added name their
   own neuron.
3. **The base lines chase the top of the stack.** Because a child is covered by its own table's bid in the
   frame it fires, the next bar's row never holds the child on the apex; it holds the child's child. So
   down's table names 6 at −1, then 10, 13, 17, 19, 22, as each new top appears, and up's table does the
   same with the other stack.
4. **Each stack dies and regrows.** "Owner, 4 at −6" earns only while 4 at −6 is uncovered. Once anything
   names 4 at −1 its previous activation is covered, the node covers nothing, the collapse drops it, the
   line is retired, 6 dies, and every line above that named 6 fails. Then 4 is uncovered again, the same
   seed recurs, and the stack rebuilds. That is the added-and-retired column.
5. **The bigrams flicker through the early rows.** With `H` 256 the first laps' rows, where every base
   symbol was bare, stay in the history for the whole run. When a bigram is retired its neighbors there
   fall back to the residual, the same seed pays on those rows, and the same line is added again a frame
   later with a new child. At lap 14 "owner, down at −1" went out and came back as a new pattern in the
   same call.
6. **Bar-specific lines do form, on the stack tops.** "Owner, 4 at −3, 4 at −1" in down's table holds at
   bar 3 alone; "owner, 5 at −3, 4 at −1" in up's table at bar 1; "owner, 4 at −5, 5 at −1" at bar 5. Each
   predicts its bar exactly while it lives, and lives as long as the children it names.

The ring described in the discussion, each bar's line naming the previous bar's child and nothing above
the children, never closes, because the self-period line in each child's table is seeded from the same
evidence as the next bar's line and bids first, in the child's own frame.

## The letter stream

`th ch sh wh` repeated four times, radius 1: `h`'s table holds one function per predecessor, each naming
`h` and the letter at −1, each with a child, and no class. Under D27 and D33 as written a function needs
one pair twice while a class needs two members twice, so each letter is named outright the second time it
comes before `h`, and the class seed only ever sees the residual. The worked case in
[algorithm-text.md](algorithm-text.md) §2 expects a class here.

`apps/text/jobs/text.js` on `apps/text/data/test.txt` (127 bytes), `H` 256, radius 1: 17% of next bytes
right on the first pass, 56% on the second, with children voting in 35% of frames.

## Running it

```bash
cd brain
cargo test --release -p brain-core                       # four behavioral tests; the cycle target is ignored
cargo run --release -p brain-core --example simple -- 200 256 6
UCAR_WINDOW=5 UCAR_LAP=1 cargo run --release -p brain-core --example simple -- 24 256 6   # tables per lap
UCAR_TRACE=1 cargo run --release -p brain-core --example simple -- 40 256 6               # adds, retires, deaths
UCAR_DUMP=1  cargo run --release -p brain-core --example simple -- 100 256 6              # every table at the end
cargo run --release -p brain-core --example text -- "th ch sh wh th ch sh wh " 1 64 h
```
