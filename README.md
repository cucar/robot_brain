# Robot Brain

A machine that compresses what it observes and acts on the best reward estimate it holds: an
implementation of **UCAR** (Universal Compression with Actions and Rewards), the algorithm specified in
[docs/algorithm.md](docs/algorithm.md).

The machine is defined by two alphabets, the events it can observe and the actions it can execute. Every
symbol, base or learned, is a **neuron**. Each frame the world reports what was observed, what ran and what
it earned; the machine compresses the frame in waves, the neurons that fired bidding the patterns of their
tables and an election taking the bids that shorten the file; what no bid covers is the **apex**, the
machine's state, and from it the machine infers the frame ahead, one winner per dimension, actions by
reward estimate and events by share of voters. There are no training epochs, no gradients and no labels:
the two free parameters are `H`, the depth of a neuron's history, and `R`, the radius each activation
dimension is seen at.

The core is Rust (`brain/brain-core`), exposed to Node.js through an N-API binding (`brain/brain-napi`),
with jobs for stocks, text and MNIST under `apps/`.

## Quick Start

**Prerequisites:** [Node.js](https://nodejs.org/) with [pnpm](https://pnpm.io/), and the
[Rust toolchain](https://rustup.rs/).

```bash
git clone https://github.com/cucar/robot_brain.git
cd robot_brain
pnpm install
```

### Build the native addon

The jobs load `brain/brain-napi/brain-napi.node`, built from the Rust core. Build it once, and again after
changing Rust code:

```bash
cd brain
./build.sh          # Linux / macOS
./build.ps1         # Windows (PowerShell)
```

### Run the cycle test

The simplest closed loop: one channel, a binary price move and own/out, a six-bar pattern repeated. The
world runs the right action wherever the machine is silent, and pays the bar's return signed by the action.

```bash
node apps/stocks/jobs/cycle.js --cycles 100 --h 256 --radius 6
```

The same test runs without Node from the Rust crate, with a trajectory line every few laps and a dump of
every table at the end:

```bash
cd brain
cargo run --release -p brain-core --example simple -- 100 256 6
UCAR_DUMP=1 cargo run --release -p brain-core --example simple -- 100 256 6
```

## Layout

| Path | What it is |
|------|------------|
| `docs/algorithm.md` | The specification: definitions **D** and rules **R**, which together are the machine. |
| `docs/algorithm-*.md` | Worked cases (text, alternation, copy, equal, hit, loop, addition, MNIST, stocks), the remarks, the evaluation and the implementation plan. |
| `brain/brain-core` | The machine in Rust, one file per type. The neuron side (Parts II and IV): `neuron` (the table, history and the `process functions` call), `pattern`, `history`, `cover` (recognition), `collapse`, `margin`, `pick`, `merge`, `bid`, `connection`. The machine side (Parts III and V): `machine` (the frame loop and waves), `activation`, `neighborhood`, `election`, `index`, `death`, `connect`, `expansion`, `selection`. `offsets` is D6 and `types` the alphabet, channels and policies. |
| `brain/brain-core/examples` | `simple` (the six-bar cycle) and `text` (a letter stream, dumping the tables asked for). |
| `brain/brain-core/tests` | Behavioral tests: a repeated pair, varied predecessors, prediction over passes, weak activations, and the cycle settling. |
| `brain/brain-napi` | The N-API binding: the `Brain` class. |
| `brain/src/index.js` | The entry point the apps import: `import { Brain } from 'brain'`. |
| `apps/stocks`, `apps/text`, `apps/mnist` | Jobs on real data, each a standalone script. `apps/db` holds MySQL utilities for backup folders of the previous core and is not used by the machine. |
| `docs/ucar-results.md` | What the implementation does on the cycle and the text stream, with the traces read. |

## The `Brain` API

```javascript
import { Brain } from 'brain';

const m = new Brain({ h: 256, timeRadius: 6, policy: 'eventsToActions', actionPolicy: 'leastBad' });
const ch = m.declareChannel({
  name: 'TEST',
  event: { name: 'price_change', resolution: 2 },
  action: { name: 'activity', resolution: 2 },
});
// a frame: [dim, bucket, p1, p2] per reported symbol, and the rewards for actions already run
const out = m.processFrame(new Int32Array([ch.eventDim, 1, 0, 0, ch.actionDim, 1, 0, 0]),
                           [{ reward: 0.9, channels: [ch.channelId], frames: 1 }]);
out.winners;   // what the machine infers for the frame ahead, one per dimension
out.stats;     // waves, bids, accepted, minted, reused, retired, added, neurons, patterns, voters
```

| Option | Default | Meaning |
|--------|---------|---------|
| `h` | 256 | The depth of every neuron's history, in its own activations (D18, R4). |
| `timeRadius` | 1 | The radius in time: how far a base neuron sees exactly (D4, D6). Spatial axes are declared with `declareAxis(name, radius)`. |
| `policy` | `all` | The neighborhood policy (D50): `all`, or `eventsToActions`, where no action is anyone's neighbor and only events and interneurons connect, to actions alone. |
| `actionPolicy` | `leastBad` | What an action dimension outputs when every candidate is negative (D51): `leastBad` or `nothing`. |

`heightCounts()`, `describeNeuron(id)`, `dumpTables(limit)` and `stats()` read the machine; `resetContext()`
drops the open activations and standing inferences and keeps the tables and connections.

## Jobs

| Job | What it runs |
|-----|--------------|
| `apps/stocks/jobs/cycle.js` | The six-bar cycle, `--cycles`, `--h`, `--radius`, `--trace`. |
| `apps/stocks/jobs/extended-cycle.js` | Twelve real bars of KGC repeated, price and volume channels. |
| `apps/stocks/jobs/stocks.js` | Daily bars over a list of symbols (`--symbols`, `--timeframe`, `--episodes`, `--holdout`), reporting how often the machine spoke and was right, and the return of holding what it owns against buy-and-hold. Data from `download-alpaca.js` or `download-yahoo.js` after `setup.js`. |
| `apps/text/jobs/text.js` | A file streamed one byte per frame (`--file`, `--episodes`, `--h`, `--radius`), scoring the inferred next byte. |
| `apps/mnist/jobs/mnist.js` | An image channel over x, y and time and a digit action channel, two frames per image (`--image-size`, `--per-class`, `--teach`, `--test`). Data from `download.js`. |

Every job imports the binding by path and takes its options on the command line; the header of each file
lists them.

## Status

The implementation follows the specification on the `wavefront` branch of the docs, with one deliberate
departure, stated at the top of `neuron.rs`: a node naming a neuron also holds where a child stands that
covers an activation of that neuron at the node's offset, and the bid then covers the child. With it the
six-bar cycle settles: every bar right and nothing added or retired from lap 111 at `H` 256 and radius 6.
A second, in `neighborhood.rs`: base activations reported in one frame are neighbors of one another, which
is what lets an image's pixels see each other.
[docs/ucar-results.md](docs/ucar-results.md) records what the machine does on the cycle and on a letter
stream, the readings that were tried, what the traces show, and what is not implemented yet.

## Documentation

- [docs/algorithm.md](docs/algorithm.md), the specification, and its companions listed in its last section.
- [docs/ucar-results.md](docs/ucar-results.md), the results and the reading of them.
- The documents under `docs/` on the previous core, its architecture, error-driven learning and the
  neuron-reuse work, are kept as history; the machine they describe was replaced by this one.

## License

Copyright 2025-2026 Cagdas Ucar. Licensed under the [Apache License 2.0](LICENSE).
