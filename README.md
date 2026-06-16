# Robot Brain

A hierarchical temporal neural network that learns patterns from raw sequential data, builds its own neuron hierarchy on demand, and makes predictions through a voting mechanism inspired by how neurons reach consensus.

No training epochs. No backpropagation. No labeled data.

You feed it streams of events — stock prices, text characters, sensor data — and it self-organizes. Neurons form, compete, decay, and die. The ones that make good predictions survive.

The brain core is implemented in Rust (with Rayon multi-threading) and exposed to Node.js via N-API. Python bindings are planned.

## How It Works

The brain is a **prediction machine**. Every neuron exists to predict what comes next. Learning happens when predictions fail.

### The Core Loop

Each frame, the brain:

1. **Observes** — receives events from input channels (prices, characters, pixels, etc.)
2. **Activates** — finds or creates neurons for the observations
3. **Recognizes** — checks if any learned patterns match the current context
4. **Learns connections** — strengthens links between co-occurring neurons
5. **Learns from errors** — when a confident prediction fails, creates a pattern to remember the context
6. **Votes** — all active neurons vote on what happens next, weighted by level and recency
7. **Acts** — executes the winning action predictions through output channels
8. **Decays** — unused connections and patterns weaken over time

### What Makes It Different

**Hierarchy emerges from failure.** When a base neuron's prediction fails, a level-1 pattern is created. When that pattern's prediction fails, a level-2 pattern is created. Abstraction isn't designed — it's earned.

**Voting enables consensus.** There's no central controller. Every active neuron contributes its prediction, weighted by its level in the hierarchy and how recently it was activated. Higher-level patterns carry more weight because they represent more context.

**Patterns override connections.** When a pattern activates on a parent neuron, it suppresses the parent's raw connection predictions. This is how the brain corrects itself — patterns exist specifically to fix prediction errors.

**Time is structural.** Temporal distance is encoded directly in connections. A connection doesn't just say "A predicts B" — it says "A predicts B at distance 3" (three frames later). This makes sequences first-class citizens.

**Multiple channels converge.** One data stream is mediocre. Many streams together is where it gets powerful — cross-modal patterns emerge naturally when multiple channels feed into the same brain.

## Quick Start

```bash
# Clone the repository
git clone https://github.com/cucar/robot_brain.git
cd robot_brain

# Install dependencies (pnpm workspace — links robot-brain into each app)
pnpm install
```

## Demo 1: Single-Channel Synthetic Cycle

A single-stock variant of the cycle test: one channel, a repeating 12-frame price/volume pattern, 20 repeats. Same idea as Demo 1 but isolated to a single channel so you can see the brain converge on optimal actions without multi-channel reinforcement doing any of the work.

```bash
node apps/stocks/jobs/synthetic-extended-test.js --error-mode static --error-threshold 0.3 --merge-threshold 0.9
```

**Expected output:**
```
Overall Optimal Rate: 233/240 = 97.1%
```

With the right thresholds the brain converges to 97%+ optimal action decisions on a single-channel cyclical pattern — confirming that hierarchy and action inference work without cross-channel consensus.

## Demo 2: Multi-Channel Synthetic Cycle

The brain learns to trade 3 stocks simultaneously (KGC, GLD, SPY), each as a separate channel. A repeating 12-day price cycle is presented 20 times — the brain discovers cross-stock patterns and converges on optimal buy/sell timing.

Run the multi-channel test with customized hyperparameters:

```bash
node apps/stocks/jobs/multi-channel-test.js --error-mode static --error-threshold 0.3 --merge-threshold 0.9
```

**Expected output:**
```
🎯 Overall Optimal Rate: 700/720 = 97.2%
```

The brain learns when to own vs. not own each stock based on upcoming price movements, achieving 96%+ optimal trade decisions across all three channels. This demonstrates how multiple input streams converge to improve inference — one of the architecture's core strengths.

## Demo 3: Stock Trading

The brain learns to trade stocks from historical price and volume data. Each stock is a separate channel — the brain discovers cross-stock patterns and makes buy/sell/hold decisions optimized by reward feedback.

**The included timeframe data is ready to use** — no API key needed for this demo.

```bash
node apps/stocks/jobs/test.js --context-length 3 --columns 20 --transaction-cost 0.02 --max-positions 10
```

**Expected output:**
```
Final Training Results (1 episodes):
============================================================
📈 Overall Performance:
   Starting Capital: $15000.00
   Total Net Profit: $1916559.50
   Average per Episode: $1916559.50
   Average ROI: +12777.06%
   Average Per-Frame ROI: +0.090457%
   Average Sharpe Ratio: 0.53
   Total Transaction Cost: $158030.95 (0.02% per trade)
   Total Trades: 71580
   Average Trades per Episode: 71580.0

💰 Net Profit & ROI by Episode:
   Episode 1: $1916559.50 | ROI: +12777.06%, +0.090457%/frame, Sharpe: 0.53 (71580 trades)

📊 Base Level Accuracy by Episode:
   Episode 1: 51.24%
```

The brain achieves ~50% base-level prediction accuracy on price movements (which is expected — markets are noisy), 
but the **reward-weighted action selection** turns that into profitable trading by learning which contexts 
produce better outcomes.

### Brain P&L over time

![Brain portfolio P&L by frame](images/brain_stocks.jpg)

Drawdowns happen, but the brain keeps making new highs — the trajectory shape a learning system should produce.

### Random Baseline Comparison

A natural worry about Demo 3 is that the result might come from a favorable data window rather than learned trading. To rule that out, the same job can be run with `--random-baseline`, which replaces the brain's action inference with a coin flip: 50% chance to be fully out, 50% chance to own one stock chosen uniformly at random. Everything else — the same data, same portfolio sizing, same execution path — is unchanged.

```bash
node apps/stocks/jobs/test.js --random-baseline
```

`--random-baseline` skips encoding, reward collection, and `brain.processFrame()` entirely, so the run is purely the trading harness driven by random signals — no shared work with the brain path.

**Two example random-baseline runs:**

![Random baseline run 1](images/random_stocks1.jpg)
![Random baseline run 2](images/random_stocks2.jpg)

Random P&L varies wildly between seeds — one run drifts up to roughly +$32K and gives most of it back to end near +$8K; another finishes in the single thousands. Across the random runs we observed, none came close to the brain's +$66K. Two differences stand out:

- **Endpoint magnitude.** The brain's final profit is roughly an order of magnitude above the best random run on this data — well outside the random baseline's natural variance.
- **Trajectory shape.** Random P&L drifts up and reverts (classic undirected exposure just riding whatever the basket does). The brain has drawdowns but keeps making new highs.

This is the simplest sanity check that the brain is doing real work: same harness, same data, only the decision source changes — and the results are not in the same league.

## Demo 4: Action Learning in Low Accuracy

The brain learns the best actions to perform in each situation over repeated episodes, even when base prediction accuracy is low.

Run the test:
```bash
node apps/stocks/jobs/test.js --symbols SO,VALE,STLD,GOOGL,MU,PLTR,UUUU,PFE,CRM,HAL --context-length 3 --columns 20 --no-summary --episodes 5
```

**Expected output:**
```
💰 Net Profit & ROI by Episode:
   Episode 1: $1572298.72 | ROI: +10481.99%, +0.086800%/frame, Sharpe: 0.52 (34016 trades)
   Episode 2: $4808085.66 | ROI: +32053.90%, +0.107505%/frame, Sharpe: 0.68 (35495 trades)
   Episode 3: $795688834.51 | ROI: +5304592.23%, +0.202679%/frame, Sharpe: 1.42 (36855 trades)
   Episode 4: $3523305600.37 | ROI: +23488704.00%, +0.230432%/frame, Sharpe: 1.65 (37662 trades)
   Episode 5: $16922477016.30 | ROI: +112816513.44%, +0.259710%/frame, Sharpe: 1.90 (37957 trades)

📊 Base Level Accuracy by Episode:
   Episode 1: 52.08%
   Episode 2: 48.32%
   Episode 3: 48.86%
   Episode 4: 48.93%
   Episode 5: 49.02%
```

## Demo 5: Stock Sequence Memorization

The brain memorizes a repeating stock price sequence across 5 episodes, reaching 95%+ prediction accuracy. This demonstrates convergence on financial data — the same learning curve seen in text memorization.

Run the stock test with customized hyperparameters for sequence memorization:

```bash
node apps/stocks/jobs/test.js --no-summary --episodes 5 --symbols KGC,GOLD,SPY --context-length 3 --forget-rate 0.0005 --error-mode static --error-threshold 0.3
```

**Expected output:**
```
🎯 Final Training Results (5 episodes):
============================================================
📈 Overall Performance:
   Starting Capital: $15000.00
   Total Net Profit: $458001313843017664.00
   Average per Episode: $91600262768603536.00
   Average ROI: +610668418457356.88%
   Average Per-Frame ROI: +0.428368%
   Average Sharpe Ratio: 3.67
   Total Trades: 67256
   Average Trades per Episode: 13451.2

💰 Net Profit & ROI by Episode:
   Episode 1: $469216.32 | ROI: +3128.11%, +0.064686%/frame, Sharpe: 0.34 (11175 trades)
   Episode 2: $117771629520472.50 | ROI: +785144196803.15%, +0.424946%/frame, Sharpe: 3.62 (14163 trades)
   Episode 3: $20027645404237436.00 | ROI: +133517636028249.58%, +0.520989%/frame, Sharpe: 4.57 (14179 trades)
   Episode 4: $192846853255107872.00 | ROI: +1285645688367386.00%, +0.563369%/frame, Sharpe: 4.88 (13941 trades)
   Episode 5: $245009043553682656.00 | ROI: +1633393623691217.75%, +0.567850%/frame, Sharpe: 4.92 (13798 trades)

📊 Base Level Accuracy by Episode:
   Episode 1: 52.47%
   Episode 2: 55.46%
   Episode 3: 58.72%
   Episode 4: 59.81%
   Episode 5: 60.33%
```

The brain goes from 50% accuracy (random) to 60% in 5 episodes on 3 stocks × 5300 frames of real market data. 
The low forget rate allows patterns to survive the full sequence. Short context (3 frames) needed because of computation limitations.

## Demo 6: Text Sequence Learning

The brain learns to predict character sequences. Feed it a string, and it memorizes the pattern — reaching ~99.94% prediction accuracy within two episodes and staying flat there.

Run the text test with customized hyperparameters for text learning (the defaults are tuned for stock data):

```bash
node apps/text/jobs/test.js --file abramov.txt --error-mode static --error-threshold 0.3 --context-length 20 --merge-threshold 0.9 --forget-rate 0.00003 --no-summary
```

**Expected output:**
```
📊 Accuracy by Episode:
   Episode 1: 27.81% (32674 frames)
   Episode 2: 99.96% (32674 frames)
   Episode 3: 99.96% (32674 frames)
   Episode 4: 99.96% (32674 frames)
   Episode 5: 99.96% (32674 frames)
```

The brain goes from low accuracy to ~99.96% in two episodes and holds there — it has fully memorized the character sequence except for the first ~20 characters at the start of each episode. Those leading characters can't be predicted because the brain hasn't seen any context yet — it needs a `context-length` window of prior characters in memory before it can recognize patterns and cast votes. The "warmup" frames at the head of each episode are a structural property of context-based prediction, not a learning failure: every character past the warmup window is predicted correctly.

## Demo 7: MNIST Digit Classification (Naive Bayes Baseline)

A sensory-only MNIST classifier built on the brain's count-based voting. One channel per pixel position (retinotopic, 784 channels at 28×28), all firing concurrently in a single frame per image. Supervision lands on a separate `digit` action channel via `brain.learn()`. With `patternForgetRate=0` and a constant reward of 1, the brain's per-voter consensus reduces to a Naive Bayes posterior — every `learn()` call increments per-pixel-per-digit counts, and inference picks the argmax.

Download the MNIST data first (one-time, ~11 MB into `apps/mnist/data/`):

```bash
node apps/mnist/jobs/download.js
```

This fetches the four standard IDX files (60k training, 10k test, gzipped) from Google's MNIST mirror. The job skips files that already exist, so it's safe to re-run.

Then run the MNIST test:

```bash
node apps/mnist/jobs/test.js --image-size 14 --buckets 2 --columns 20 --per-class 0 --max-test-images 0 --episodes 1 --error-mode static --error-threshold 0.1 --merge-threshold 0.9
```

**Expected output:**
```
MNIST — sensory-only (Naive Bayes) baseline
  Image size: 14×14 (196 channels)
  Buckets: 2 (Phase A — binary)
  Context length: 1
  Forget rate: 0
  Consensus: nb
  Episodes: 1
  Training: balanced,
  Test images: all

  Balanced training set built: 5421 per class × 10 = 54210 total
  Episode 1/1: train=92.85% (50332/54210) | 130 img/s 418150ms | 0:96% 1:98% 2:93% 3:93% 4:92% 5:86% 6:96% 7:94% 8:89% 9:91%
    ↳ spatial: depth=1 | L1:31834 | 31834 active, 31834 minted cum | 32202 neurons

  Test: 94.88% (9488/10000) 51638ms | 0:98% 1:98% 2:94% 3:95% 4:95% 5:94% 6:96% 7:92% 8:93% 9:93%

Results
======================================================================
  Episode 1: train=92.85% (418150ms)

  Spatial hierarchy (after each episode):
    Episode 1: depth=1 | L1:31834 | 31834 active, 31834 minted cum | 32202 neurons
  Test:     94.88% (9488/10000)

  Confusion (rows = actual, cols = predicted):
           0    1    2    3    4    5    6    7    8    9
   0     960    0    0    1    0    5    6    1    7    0
   1       0 1116    3    3    0    0    5    0    8    0
   2      10    1  971    7    7    1    2    8   24    1
   3       1    0   11  955    0   15    0   12   11    5
   4       0    2    2    0  935    0   10    0    3   30
   5       5    3    0   20    1  840    8    2    8    5
   6       7    4    1    0    4   16  919    0    7    0
   7       1   17   17    2    5    0    0  944    7   35
   8      11    0    3   16    5    8    5    6  910   10
   9      10    7    2    7   18    6    0   10   11  938
======================================================================

⏱️  Total Execution Time: 7m 50s
```

94.88% test accuracy from a single training pass. The confusion matrix errors are digits whose pixel-level marginals overlap heavily. 
Class-balanced training is helps here - when trained on full data, the result is 94.56%.

### Downloading Fresh Stock Data

To download new data or different timeframes, you need a free [Alpaca](https://alpaca.markets) account:

1. Sign up at [alpaca.markets](https://alpaca.markets) (free paper trading account)
2. Get your API key and secret from the dashboard
3. Copy `apps/stocks/.env.example` to `apps/stocks/.env` and fill in your credentials:
   ```
   ALPACA_KEY_ID=your_key_here
   ALPACA_SECRET_KEY=your_secret_here
   ```
4. Download data (Alpaca, or `download-yahoo.js` for longer daily history):
   ```bash
   node apps/stocks/jobs/download-alpaca.js --timeframe 3H
   ```
5. Process downloaded data into training files:
   ```bash
   node apps/stocks/jobs/setup.js --timeframe 3H
   ```
6. Run the training job:
   ```bash
   node apps/stocks/jobs/test.js --timeframe 3H
   ```

## Architecture

```mermaid
graph TB
    subgraph Brain["🧠 Brain"]
        direction TB
        subgraph Components[" "]
            direction LR
            TH["<b>Thalamus</b><br/>neuron registry<br/>channel mgmt<br/>fast lookup"]
            MEM["<b>Memory</b><br/>active neurons<br/>inferred neurons<br/>sliding window"]
            NEU["<b>Neuron</b><br/>connections<br/>children (patterns)<br/>voting & learning<br/>lazy decay"]
            CTX["<b>Context</b><br/>pattern entries<br/>threshold matching<br/>merge logic"]
        end
        subgraph Pipeline["Frame Processing Pipeline"]
            direction LR
            P1["getFrame"] --> P2["age"] --> P3["activate"] --> P4["recognize<br/>patterns"]
            P4 --> P5["learn<br/>connections"] --> P6["learn from<br/>errors"] --> P7["vote &<br/>infer"]
            P7 --> P8["execute<br/>actions"] --> P9["decay"]
        end
    end
    CH1["📈 Stock"] -- "events →" --> Brain
    CH2["📝 Text"] -- "events →" --> Brain
    CH3["👁 Vision"] -- "events →" --> Brain
    CH4["🔊 Audio"] -- "events →" --> Brain
    Brain -- "→ actions" --> CH1
    Brain -- "→ actions" --> CH2
    Brain -- "→ actions" --> CH3
    Brain -- "→ actions" --> CH4
```

### How Hierarchy Emerges

```mermaid
graph BT
    subgraph L0["Level 0 — Base Neurons"]
        A["A (event)"]
        B["B (event)"]
        C["C (event)"]
        E["E (event)"]
    end
    subgraph L1["Level 1 — Patterns correct base errors"]
        P1["Pattern₁<br/>parent: B<br/>context: A@2, D@1<br/>predicts: E"]
    end
    subgraph L2["Level 2 — Patterns correct pattern errors"]
        P2["Pattern₂<br/>parent: Pattern₁<br/>context: Pattern₀@3<br/>predicts: C"]
    end
    A -- "dist=2" --> B
    B -- "dist=1" --> C
    B -- "dist=1" --> E
    B -. "predicted C, got E → create" .-> P1
    P1 -. "predicted E, got C → create" .-> P2
```

### Core Components

The brain core is a Rust workspace (`brain/`) with two crates:

| Crate / File | Role | Description |
|------|------|-------------|
| `brain-core/src/brain.rs` | Orchestrator | Frame processing loop, pattern recognition, learning, inference |
| `brain-core/src/thalamus.rs` | Relay station | Neuron registry, channel management, dimension mappings, quantizer |
| `brain-core/src/region.rs` | Parallelism | Column partitioner, Rayon-based multi-threaded dispatch |
| `brain-core/src/column.rs` | Worker | Owns a neuron partition, batch operations (becomes a thread in multi-column mode) |
| `brain-core/src/memory.rs` | Short-term memory | Temporal sliding window of active neurons indexed by age |
| `brain-core/src/neuron.rs` | Neuron | Connections, routing table, voting, learning, lazy decay |
| `brain-core/src/context.rs` | Pattern context | Context representation, threshold-based matching, merge logic |
| `brain-core/src/quantizer.rs` | Quantization | Scalar-to-bucket discretization (static, dynamic, passthrough) |
| `brain-core/src/backup.rs` | Persistence | File-based backup/restore (CSVs under `<jobDir>/backups/<label>/`) |
| `brain-core/src/diagnostics.rs` | Metrics | Accuracy tracking and continuous error measurement |
| `brain-napi/` | N-API bridge | Exposes Rust Brain as a native Node.js addon |
| `libs/node` | Node bindings | Re-exports the native addon + Job runner; published to npm as `robot-brain` |

### Apps

Each app owns an encoder (and optionally a trader) that describes its channels to the brain via a spec (`registerChannelSpec`). The spec lists the channel's dimensions, their bucket resolutions, and whether each dim is an input (event) or output (action). Base neurons carry exactly one `(dimId, bucketId)` pair — multi-dim observations emit multiple base neurons per frame.

| App | Inputs (Events) | Outputs (Actions) | Reward Signal |
|-----|-----------------|-------------------|---------------|
| `apps/stocks` | One neuron per dim: price change, volume change | One neuron: position (own/out) | Profit/loss |
| `apps/text`   | One neuron: character code | — | — |
| `apps/db`     | MySQL utilities (import/export) — not a brain channel; loads/exports backup folders for analysis | — | — |

### Jobs

Jobs define learning scenarios — which encoders to register, how to configure them, and how to run episodes:

| Job | Description |
|-----|-------------|
| `apps/stocks/jobs/test.js` | Multi-stock trading with historical data |
| `apps/stocks/jobs/multi-channel-test.js` | Multi-symbol trading across shared brain |
| `apps/stocks/jobs/synthetic-cycle-test.js` | Cycle-learning synthetic stress test |
| `apps/stocks/jobs/synthetic-extended-test.js` | Extended cycle synthetic with optimality analysis |
| `apps/text/jobs/test.js` | Character sequence memorization (default `data/test.txt`; override with `--file`) |
| `apps/db/import.js` | Bulk-load a backup folder into MySQL via `LOAD DATA LOCAL INFILE` |
| `apps/db/export.js` | Dump current MySQL state to `./backups/<timestamp>/` in cwd |

## Hyperparameters

All hyperparameters are configured via the Brain constructor options and can be passed as command-line arguments:

| Parameter | Default | Command Line Option | Description |
|-----------|---------|---------------------|-------------|
| `errorCorrectionMode` | `'conservative'` | `--error-mode` | Threshold function for creating correction patterns: `static` (fixed), `conservative` (mean + σ — learn outliers), `neutral` (mean), `aggressive` (mean − σ — memorize aggressively). Per-(neuron, age) error rate stats are tracked online via Welford's algorithm. |
| `errorCorrectionThreshold` | 0.5 | `--error-threshold` | When `errorCorrectionMode='static'`, the fixed prediction error threshold. For dynamic modes, the warmup fallback used until 3 samples have been observed at a given (neuron, age) pair. |
| `contextLength` | 10 | `--context-length` | Frames a neuron stays active in the sliding window |
| `mergeThreshold` | 0.5 | `--merge-threshold` | Min context match ratio for pattern recognition |
| `patternForgetRate` | 0.01 | `--forget-rate` | Pattern prediction decay rate per frame |

## Command Line Options

```bash
node <path-to-job.js> [options]
```

| Option | Description |
|--------|-------------|
| `--timeframe <tf>` | Data timeframe for stock jobs (e.g., `1D`, `1H`, `3H`, `1Min`) |
| `--episodes <n>` | Number of training episodes |
| `--holdout <n>` | Hold out last N rows from training |
| `--offset <n>` | Skip first N rows |
| `--symbols <list>` | Comma-separated list of stock tickers (e.g. `KGC,GLD,SPY`) |
| `--max-positions <n>`| Maximum number of stock positions to hold at once |
| `--max-price <n>` | Maximum price limit for stocks |
| `--initial-capital <n>`| Starting capital for the portfolio |
| `--transaction-cost <n>`| Simulated transaction cost per trade, as a percentage (e.g. `0.01` = 0.01%). Buys pay more, sells receive less. Reports total cost at end of run |
| `--context-length <n>`| Sliding window size (frames) |
| `--forget-rate <n>` | Pattern activation decay rate per frame |
| `--error-mode <m>` | Error-correction threshold mode: `static`, `conservative`, `neutral`, `aggressive` |
| `--error-threshold <n>`| Static threshold value (when mode=`static`); warmup fallback for dynamic modes |
| `--merge-threshold <n>`| Threshold for pattern context matching |
| `--debug` | Show detailed frame-by-frame processing |
| `--diagnostic` | Show inference and conflict resolution details |
| `--save-brain <label>` | Save a CSV backup on shutdown (incl. crash) under `<jobDir>/backups/<label>/` |
| `--load-brain <label>` | Load a labeled backup before the first frame (errors if none exists) |
| `--save-context <label>` | Save the memory context window on shutdown under `<jobDir>/contexts/<label>/` |
| `--load-context <label>` | Restore the memory context window (active neurons, votes, rewards) |
| `--save-session <label>` | Save trader/portfolio state on shutdown under `<jobDir>/sessions/<label>/` |
| `--load-session <label>` | Restore trader/portfolio state (positions, cash, prices) |
| `--no-summary` | Suppress per-frame summary output |
| `--start <date>` | Start date for data (YYYY-MM-DD) |
| `--end <date>` | End date for data (YYYY-MM-DD) |
| `--random-baseline` | Skip the brain entirely; pick own/out + symbol uniformly at random (sanity-check baseline for stock test) |

## Creating Custom Jobs

```javascript
import { Job, runJob } from 'robot-brain';
import { TextEncoder } from '../encoder.js';

export default class MyJob extends Job {

    constructor() {
        super();
        this.encoders = [];
    }

    async initialize() {
        const encoder = new TextEncoder('text');
        const ids = this.brain.registerChannelSpec(encoder.getChannelSpec());
        encoder.bindIds(ids);
        this.encoders.push(encoder);
    }

    async configureChannels() {
        for (const encoder of this.encoders) encoder.setData('hello world');
    }

    async executeJob() {
        this.brain.resetContext();
        while (true) {
            const inputs = new Map();
            let any = false;
            for (const encoder of this.encoders) {
                const frame = encoder.nextFrame();
                if (!frame) continue;
                any = true;
                inputs.set(encoder.channelId, encoder.encode(frame));
            }
            if (!any) break;
            this.brain.processFrame(inputs, new Map());
        }
    }

    async showResults() {
        console.log(this.brain.getEpisodeSummary());
    }
}

await runJob(import.meta, MyJob);
```

Save as `apps/text/jobs/my-job.js` and run with `node apps/text/jobs/my-job.js`.

## Documentation

- **[Architecture Design](docs/architecture.md)** — detailed design document covering voting, patterns, frame processing, and data structures
- **[Error-Driven Learning](docs/error-driven-learning.md)** — deep dive on how patterns are created from prediction errors
- **[Technical Foundations](docs/TECHNICAL_FOUNDATIONS.md)** — architectural ideas, biological inspirations, and comparison with conventional approaches
- **[Hippocampus Design](docs/hippocampus.md)** — design and implementation plan for the hippocampal region (long-term memory, thinking, metacognition)
- **[Future Work](docs/future-work.md)** — MNIST benchmarks, Python bindings, MPI distribution, and other planned work

## Persistence

The brain runs entirely in-memory. Three labeled save/load pairs let you snapshot
and resume across sessions:

| Pair | What it persists | Storage path |
|------|-----------------|--------------|
| `--save-brain` / `--load-brain` | Learned neurons and connections | `<jobDir>/backups/<label>/` |
| `--save-context` / `--load-context` | Memory context window (active neurons, votes, rewards) | `<jobDir>/contexts/<label>/` |
| `--save-session` / `--load-session` | Trader/portfolio state (positions, cash, prices) | `<jobDir>/sessions/<label>/` |

```bash
# Run the first half and save everything
node apps/stocks/jobs/test.js --episodes 1 --frames 1250 --save-brain day1 --save-context day1 --save-session day1

# Resume from where we left off
node apps/stocks/jobs/test.js --episodes 1 --offset 1251 --load-brain day1 --load-context day1 --load-session day1
```

Each save is a folder of CSVs under the label you choose. Brain backups are also
written on crash (uncaught error / SIGINT) when `--save-brain` is set.

For MySQL-based analysis tooling — bulk-loading a backup into a queryable database
or exporting MySQL state back to a backup folder — see the [`apps/db`](apps/db)
app. It is not part of the brain core; the brain has no DB dependency.

### Backup → MySQL → Backup Round-Trip

Take a backup, push it through MySQL, pull it back out, and verify the rehydrated
brain reproduces the same result. Uses the [Demo 6](#demo-6-stock-sequence-memorization)
config (KGC,GLD,SPY) — a single episode here ends around `$22,675,481.59`.

```bash
# 1. Run one episode and save a backup
node apps/stocks/jobs/test.js --no-summary --symbols KGC,GOLD,SPY --context-length 3 --error-mode static --error-threshold 0.3 --forget-rate 0.001 --save-brain roundtrip

# 2. Import that backup folder into MySQL
node apps/db/import.js apps/stocks/jobs/test/backups/roundtrip

# 3. Delete the original backup, then export MySQL back to the same label
Remove-Item -Recurse -Force apps/stocks/jobs/test/backups/roundtrip
node apps/db/export.js apps/stocks/jobs/test/backups/roundtrip

# 4. Load the round-tripped backup and run another episode — should reach
#    ~$8,441,629.32, matching what a continuous two-episode run produces
node apps/stocks/jobs/test.js --no-summary --symbols KGC,GOLD,SPY --context-length 3 --error-mode static --error-threshold 0.3 --forget-rate 0.001 --load-brain roundtrip
```

The `apps/db` import uses `LOAD DATA LOCAL INFILE`, which needs `local_infile=ON`
server-side; the import script enables it automatically (`SET GLOBAL local_infile = 1`)
as long as the connecting user has `SYSTEM_VARIABLES_ADMIN` (or `SUPER` on older
MySQL) — root has this by default. DB credentials live in [`apps/db/.env`](apps/db/.env.example).

## License

Copyright 2025-2026 Cagdas Ucar. Licensed under the [Apache License 2.0](LICENSE).
