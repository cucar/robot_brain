# Robot Brain

A hierarchical temporal neural network that learns patterns from raw sequential data, builds its own neuron hierarchy on demand, and makes predictions through a voting mechanism inspired by how cortical columns reach consensus.

No training epochs. No backpropagation. No labeled data.

You feed it streams of events — stock prices, text characters, sensor data — and it self-organizes. Neurons form, compete, decay, and die. The ones that make good predictions survive.

This is the Node.js reference implementation. A high-performance C++ core with Python and Node.js bindings is in development.

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

# Install dependencies
npm install
```

## Demo 1: Stock Trading

The brain learns to trade stocks from historical price and volume data. Each stock is a separate channel — the brain discovers cross-stock patterns and makes buy/sell/hold decisions optimized by reward feedback.

**The included 3-hour timeframe data is ready to use** — no API key needed for this demo.

```bash
node run-brain.js stock-test --timeframe 3H
```

**Expected output:**
```
Final Training Results (1 episodes):
============================================================
📈 Overall Performance:
   Starting Capital: $15000.00
   Total Net Profit: $146741.60
   Average per Episode: $146741.60
   Average ROI: +978.28%
   Average Per-Frame ROI: +0.094973%
   Total Trades: 3060
   Average Trades per Episode: 3060.0

💰 Net Profit & ROI by Episode:
   Episode 1: $146741.60 | ROI: +978.28%, +0.094973%/frame (3060 trades)

📊 Base Level Accuracy by Episode:
   Episode 1: 56.13%
```

The brain achieves ~50% base-level prediction accuracy on price movements (which is expected — markets are noisy), but the **reward-weighted action selection** turns that into profitable trading by learning which contexts produce better outcomes.

### Downloading Fresh Stock Data

To download new data or different timeframes, you need a free [Alpaca](https://alpaca.markets) account:

1. Sign up at [alpaca.markets](https://alpaca.markets) (free paper trading account)
2. Get your API key and secret from the dashboard
3. Copy `.env.example` to `.env` and fill in your credentials:
   ```
   ALPACA_KEY_ID=your_key_here
   ALPACA_SECRET_KEY=your_secret_here
   ```
4. Download data:
   ```bash
   node stock-download.js --timeframe=3H
   ```
5. Process and run:
   ```bash
   node run-setup.js stock-test --timeframe 3H
   node run-brain.js stock-test --timeframe 3H
   ```

## Demo 2: Stock Sequence Memorization

The brain memorizes a repeating stock price sequence across 5 episodes, reaching 95%+ prediction accuracy. This demonstrates convergence on financial data — the same learning curve seen in text memorization.

**Before running**, adjust the hyperparameters for stock memorization:

In `jobs/stock-test.js`, use only 3 stocks:
```javascript
symbols: ['KGC', 'GLD', 'SPY'],
```

In `brain/memory.js`, change `contextLength` to `3`:
```javascript
this.contextLength = 3;
```

In `brain/neuron.js`, change the forget rates to `0.0001`:
```javascript
static connectionForgetRate = 0.0001;
static contextForgetRate = 0.0001;
static patternForgetRate = 0.0001;
```

Then run:
```bash
node run-brain.js stock-test --timeframe 3H --episodes 5 --no-summary
```

**Expected output:**
```
🎯 Final Training Results (5 episodes):
============================================================
📈 Overall Performance:
   Starting Capital: $15000.00
   Total Net Profit: $2386888285524.23
   Average per Episode: $477377657104.85
   Average ROI: +3182517714.03%
   Average Per-Frame ROI: +0.476526%
   Total Trades: 12195
   Average Trades per Episode: 2439.0

💰 Net Profit & ROI by Episode:
   Episode 1: $3518.06 | ROI: +23.45%, +0.008411%/frame (2063 trades)
   Episode 2: $67967449.57 | ROI: +453116.33%, +0.336651%/frame (2484 trades)
   Episode 3: $37952932525.14 | ROI: +253019550.17%, +0.590311%/frame (2532 trades)
   Episode 4: $643220132782.73 | ROI: +4288134218.55%, +0.704021%/frame (2548 trades)
   Episode 5: $1705647249248.74 | ROI: +11370981661.66%, +0.743234%/frame (2568 trades)

📊 Base Level Accuracy by Episode:
   Episode 1: 50.50%
   Episode 2: 69.35%
   Episode 3: 84.32%
   Episode 4: 91.91%
   Episode 5: 95.38%
```

The brain goes from 50% accuracy (random) to 95%+ in 5 episodes on 3 stocks × 2505 frames of real market data. With more episodes it continues climbing toward 99%+. The low forget rate (0.0001) allows patterns to survive the full 2505-frame sequence, and the short context (3 frames) reduces noise from coincidental connections.

> **Remember to change the hyperparameters back** to their stock defaults (`contextLength = 20`, forget rates = `0.009`/`0.009`/`0.011`, all 10 stocks uncommented) if you want to run the default stock test afterward.

## Demo 3: Text Sequence Learning

The brain learns to predict character sequences. Feed it a string, and it memorizes the pattern — reaching 100% prediction accuracy within a few episodes.

**Before running**, adjust the hyperparameters for text learning (the defaults are tuned for stock data):

In `brain/context.js`, change `mergeThreshold` to `0.8`:
```javascript
static mergeThreshold = 0.8;
```

In `brain/neuron.js`, change the forget rates to `0.001`:
```javascript
static connectionForgetRate = 0.001;
static contextForgetRate = 0.001;
static patternForgetRate = 0.001;
```

Then run:
```bash
node run-brain.js text-test
```

**Expected output:**
```
Accuracy by Episode:
   Episode 1: 23.58% (127 frames)
   Episode 2: 89.60% (127 frames)
   Episode 3: 98.40% (127 frames)
   Episode 4: 98.40% (127 frames)
   Episode 5: 100.00% (127 frames)
```

The brain goes from ~24% accuracy (random) to 100% in 5 episodes — it has fully memorized the character sequence and can predict every next character correctly.

> **Remember to change the hyperparameters back** to their stock defaults (`mergeThreshold = 0.5`, forget rates = `0.009`/`0.009`/`0.011`) if you want to run stock tests afterward.

## Demo 4: Synthetic Cycle Memorization

The brain learns to trade 3 stocks simultaneously (KGC, GLD, SPY), each as a separate channel. A repeating 12-day price cycle is presented 20 times — the brain discovers cross-stock patterns and converges on optimal buy/sell timing.

```bash
node run-brain.js multi-channel-test
```

**Expected output:**
```
🎯 Overall Optimal Rate: 93%+
```

The brain learns when to own vs. not own each stock based on upcoming price movements, achieving 93%+ optimal trade decisions across all three channels. This demonstrates how multiple input streams converge to improve inference — one of the architecture's core strengths.

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

| File | Role | Description |
|------|------|-------------|
| `brain/brain.js` | Orchestrator | Frame processing loop, pattern recognition, learning, inference |
| `brain/thalamus.js` | Relay station | Neuron registry, channel management, dimension mappings |
| `brain/memory.js` | Short-term memory | Temporal sliding window of active neurons indexed by age |
| `brain/neuron.js` | Neuron | Connections, patterns, voting, learning, lazy decay |
| `brain/context.js` | Pattern context | Context representation, threshold-based matching, merge logic |
| `brain/database.js` | Persistence | Optional MySQL backup/restore (not used during processing) |
| `brain/diagnostics.js` | Metrics | Performance tracking and debug output |
| `brain/dump.js` | Debugging | Brain state dumps |

### Channels

Channels are adapters between the brain and external data. Each channel defines its input dimensions (events) and output dimensions (actions):

| Channel | Inputs (Events) | Outputs (Actions) | Reward Signal |
|---------|-----------------|-------------------|---------------|
| `StockChannel` | Price change, volume change, position | Buy, sell, hold | Profit/loss |
| `TextChannel` | Character code | Next character | Prediction accuracy |
| `VisionChannel` | x, y, r, g, b | Saccade direction | Target acquisition |
| `AudioChannel` | Frequency bands | — | — |
| `ArmChannel` | Joint positions, touch | Muscle contractions | Goal reaching |
| `TongueChannel` | Taste dimensions | Tongue movements | — |

### Jobs

Jobs define learning scenarios — which channels to use, how to configure them, and how to run episodes:

| Job | Description |
|-----|-------------|
| `stock-test` | Multi-stock trading with historical data |
| `text-test` | Character sequence memorization |
| `vision1` | Visual pattern learning with saccadic eye movements |
| `arm1` | Motor control with proprioceptive feedback |
| `multisensory1` | Multi-channel integration |

## Hyperparameters

All hyperparameters are configured as static properties on their respective classes:

| Parameter | Default | Class | Description |
|-----------|---------|-------|-------------|
| `contextLength` | 20 | Memory | Frames a neuron stays active in the sliding window |
| `maxStrength` | 100 | Neuron/Context | Maximum connection/pattern strength |
| `levelVoteMultiplier` | 4.25 | Neuron | Vote weight increase per pattern level |
| `rewardSmoothing` | 0.8 | Neuron | Exponential smoothing factor for reward updates |
| `eventErrorMinStrength` | 1 | Neuron | Min prediction strength to trigger error pattern creation |
| `actionRegretMinStrength` | 3 | Neuron | Min action strength to trigger regret pattern creation |
| `actionRegretMinPain` | 0 | Neuron | Min negative reward to trigger action regret |
| `mergeThreshold` | 0.5 | Context | Min context match ratio for pattern recognition (0.8 for text) |
| `negativeReinforcement` | 0.1 | Context | Weakening rate for missing context entries |
| `connectionForgetRate` | 0.009 | Neuron | Connection strength decay rate per frame (0.001 for text) |
| `contextForgetRate` | 0.009 | Neuron | Pattern context decay rate per frame (0.001 for text) |
| `patternForgetRate` | 0.011 | Neuron | Pattern prediction decay rate per frame (0.001 for text) |

## Command Line Options

```bash
node run-brain.js <job-name> [options]
```

| Option | Description |
|--------|-------------|
| `--timeframe <tf>` | Data timeframe for stock jobs (e.g., `1D`, `1H`, `3H`, `1Min`) |
| `--episodes <n>` | Number of training episodes |
| `--holdout <n>` | Hold out last N rows from training |
| `--offset <n>` | Skip first N rows |
| `--debug` | Show detailed frame-by-frame processing |
| `--diagnostic` | Show inference and conflict resolution details |
| `--database` | Enable MySQL backup/restore |
| `--no-summary` | Suppress per-frame summary output |
| `--start <date>` | Start date for data (YYYY-MM-DD) |
| `--end <date>` | End date for data (YYYY-MM-DD) |

## Creating Custom Jobs

```javascript
import { Job } from './jobs/job.js';
import { TextChannel } from './channels/text.js';

export default class MyJob extends Job {

    getChannels() {
        return [{ name: 'text', channelClass: TextChannel }];
    }

    async configureChannels() {
        this.brain.getChannel('text').setTraining('hello world', 3);
    }

    async executeJob() {
        this.brain.resetContext();
        while (await this.brain.processFrame()) {}
    }

    async showResults() {
        console.log(this.brain.getEpisodeSummary());
    }
}
```

Save as `jobs/my-job.js` and run with `node run-brain.js my-job`.

## Documentation

- **[Architecture Design](docs/architecture.md)** — detailed design document covering voting, patterns, frame processing, and data structures
- **[Error-Driven Learning](docs/error-driven-learning.md)** — deep dive on how patterns are created from prediction errors
- **[Technical Foundations](docs/TECHNICAL_FOUNDATIONS.md)** — architectural ideas, biological inspirations, and comparison with conventional approaches

## Optional: MySQL Persistence

The brain runs entirely in-memory. MySQL is optional — used only for saving/restoring brain state between sessions.

```bash
# Apply schema (requires MySQL running)
mysql -u root -p < db/db.sql

# Run with database backup enabled
node run-brain.js stock-test --timeframe 3H --database
```

## License

Copyright 2025-2026 Cagdas Ucar. Licensed under the [Apache License 2.0](LICENSE).
