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

D22 says a node naming a neuron holds where that neuron stands at the node's offset. Here it also holds
where a child stands that covers an activation of that neuron at that offset: the child's expansion
(R28) is read at the row, as base symbols at their offsets from the owner. Matching, seeding (D47) and
the collapse (D27) all read a node this way, and the bid then covers the child instead of the base
activation under it. A pattern's parent may be matched by its child. The rows themselves record the
apex alone, as D18 says.

A second departure is on the machine's side. D5 admits a same-frame neighbor only from an earlier wave,
so the base activations reported in one frame never see one another. Here they do, within reach. On the
cycle and the letter stream this changes nothing, since each frame there reports one event and one
action and the policy keeps those apart. On MNIST it is the difference between learning nothing and
learning something (below).

An earlier departure, rows recording every activation within reach with a flag for the apex, is gone.
Of all 64 combinations of the six readings that were tried (below), the only one that settles the
cycle at every `H` and radius tried is this one with nothing else changed.

## Not implemented, or simplified

- The match (R49): standing inferences are not read back against the frames they point into and are
  never struck; they stand until the window closes.
- Functions of one cover that share a variable go as separate bids, not as one (D31).
- Keys for the index are recomputed for every living interneuron after each frame; the window scan of the
  earlier implementation is gone, but nothing is incremental yet.
- Spans are per activation (D4), and the owner's reach in prices (D13) is read from the largest span any
  activation of the neuron has had.
- No backup or persistence, and no parallel dispatch: neurons are called one at a time per wave.

## The six-bar cycle

The configuration of the specification: one channel, a binary price move and own/out, radius 6, `H` 256,
the `eventsToActions` policy, the world running the right action where the machine is silent, the reward
the bar's return signed by the action, scoped to the channel and the frame.

| laps | right per bar, of 50 | added | retired | alive neurons | patterns |
|------|----------------------|-------|---------|---------------|----------|
| 1-50 | 45 28 47 48 47 50 | 18 | 5 | 13 | 11 |
| 51-100 | 50 50 50 50 50 50 | 14 | 14 | 13 | 10 |
| 101-150 | 50 50 50 50 50 50 | 1 | 4 | 11 | 7 |
| 151-2000 | 50 50 50 50 50 50 | 0 | 0 | 11 | 7 |

Every bar right from lap 31 on in windows of ten, and nothing added or retired from lap 111 to lap 2000:
seven lines, eleven neurons, one child per bar. Over 1000 laps at other settings:

| `H` | radius | every bar right from lap | nothing added or retired from lap |
|-----|--------|--------------------------|-----------------------------------|
| 64 | 3 | 61 | 81 |
| 64 | 6 | 61 | 81 |
| 64 | 12 | 151 | 151 |
| 256 | 3 | 161 | 211 |
| 256 | 6 | 31 | 111 |
| 256 | 12 | 321 | 401 |

The settled tables are a ring. Down's table holds "owner, the bar-1 child at −1" (bar 2) and "owner,
the bar-2 child at −1" (bar 3); up's table holds the bar-3 line, and bars 5, 6 and 1 are lines naming
the previous bar's child beside that lap's base symbols. One child's table holds one line, "owner, down
at −7". Each child connects to the action of the bar after it, with strength in the hundreds.

The `height` of an activation counts the chain of children covering children, which under this reading
grows by one every frame; it no longer measures the structure, and the height and levels printouts of
the examples and jobs mean nothing on the cycle.

## The readings that were tried

Six readings were switches while this was measured, and are removed now. Rows of the apex alone (D18)
or of every activation within reach with a flag; a parent matched through the child covering it (the
departure above); no-self (a function naming the owner may not name the owner's own neuron); frozen
(no re-centering, D29); strict (the collapse takes only nodes holding in every use); owner-param (the
owner's activation as a parameter position). All 64 combinations, 300 laps each, after the fixes below:

- 3 combinations were all right in laps 251-300 with nothing turning over: apex rows with through
  children alone, with no-self added, and with owner-param, strict and frozen added.
- 19 froze into a wrong structure: nothing added or retired, one to three bars wrong every lap.
- 42 kept churning, 57% to 95% right in laps 251-300. All 32 combinations without through children are
  among them, at 85% to 95%.

Over 1000 laps at `H` 64 and 256 and radius 3, 6 and 12, through children alone settled at all six
settings; with no-self it failed at three of them, and with owner-param, strict and frozen at four,
those freezing at 50 of 60 per ten laps.

## The churn, read, and four faults in the implementation

Before the switch was settled, the added-and-retired counts were read line by line (300 laps,
`UCAR_TRACE=1`, a reason printed for every retirement). Most of them were faults of the implementation,
fixed in place:

1. **Classes built and retired every frame.** A class was priced by the credit at its places less the
   choice, while the cover never took a class alone, so with no function naming it its margin was about
   −11 bits: retired the next call, rebuilt the call after. 572 of 587 retirements in laps 101-150 were
   these. A variable that no function names at the end of the create step is no longer kept (D30: over
   one variable and nothing else a function is never built), and a class alone is offered to the cover
   as D41 says.
2. **Value neurons leaked.** A member dropped from a variable never released its value neuron, which kept
   its parent entry and never reached the ledger (R38). Alive neurons grew from 18 to 181 over 3000 laps
   with 24 patterns, and since a node's price is `log₂` of the alive count (D13), every line grew dearer
   with the run. Dropped members now hand their value neurons to the ledger.
3. **Mutually redundant lines retired together.** Every line's margin was measured against the whole
   table and every negative one removed in one pass. Two lines covering the same rows each show no margin
   while the other stands, so both went and one was rebuilt. The worst goes now, and the rest are
   measured again without it.
4. **A function left with no nodes crashed recognition.** The collapse can strip a function to nothing;
   recognition ran before retirement and indexed its first node. Such a function now covers nothing until
   retirement removes it.

**What the remaining churn was, under the specification's reading.** Each bar's cheapest line is "owner,
the previous bar's newest child at −1". Every lap adds a link, so the structure is a helix: the child for
bar 2 this lap is "down after this lap's up-child", a new neuron one lap longer. That is T13 as written:
on input that repeats exactly the stack grows a level each time round and nothing caps it. Each new link
makes the older lines over the same bar redundant; their margins (D30) sit at the knife edge (a benefit
of 22.0 against a line of 23.9 after 40 uses), the roots retire, their children die, every line naming
them loses a node, and the helix rebuilds from the base lines. 3000 laps of that reading: 90.7%, never
settled. A link that names "the up-child of bar 1" through the up it covers, rather than this lap's
particular child, is the same line every lap, and the ring closes.

## The letter stream

`th ch sh wh` repeated four times, radius 1: `h`'s table holds one function per predecessor, each naming
`h`, the letter at −1 and the child covering it, each with a child, and no class. Under D27 and D33 as
written a function needs one pair twice while a class needs two members twice, so each letter is named
outright the second time it comes before `h`. The worked case in [algorithm-text.md](algorithm-text.md)
§2 expects a class here.

`apps/text/jobs/text.js` on `apps/text/data/test.txt` (127 bytes), `H` 256, radius 1: 20% of next bytes
right on the first pass, 78% on the second, with children voting in 83% of frames of the second.

## MNIST

`apps/mnist/jobs/mnist.js` at its defaults: 14×14 images, ink above 128 as one event, 1000 training
images (100 per digit, the first 200 taught, the rest rewarded ±1), 500 test images, `H` 2048, radius 1.

| neighbors in the image frame | train correct | test correct | patterns | neurons | ms per image |
|------------------------------|---------------|--------------|----------|---------|--------------|
| earlier waves only (D5) | 9.9% | 9.8% | 0 | 12 | 0.1 |
| base activations of the frame | 17.6% | 15.8% | 390 after training, 555 after test | 266, then 365 | 293 train, 411 test |

Under D5 every pixel of an image fires in the same wave and the previous image is out of reach, so no ink
activation ever has a neighbor and nothing is built; the machine says a digit from the bare ink neuron's
connections, at chance. With same-frame base neighbors pairs and corners form (254 neurons at the base,
10 one level up, 2 two levels up at the end of training), accuracy is near 17% from image 100 on and does
not climb, and the tables keep growing through the test images. The "apex/image" figure the job prints
reads the frame one reach back, the digit frame in this protocol, and is not the image's apex.

## Running it

```bash
cd brain
cargo test --release -p brain-core                       # five behavioral tests, the cycle among them
cargo run --release -p brain-core --example simple -- 200 256 6
UCAR_WINDOW=5 UCAR_LAP=1 cargo run --release -p brain-core --example simple -- 24 256 6   # tables per lap
UCAR_TRACE=1 cargo run --release -p brain-core --example simple -- 40 256 6               # adds, retires with reasons, deaths, cascades
UCAR_DUMP=1  cargo run --release -p brain-core --example simple -- 100 256 6              # every table at the end
cargo run --release -p brain-core --example text -- "th ch sh wh th ch sh wh " 1 64 h
```
