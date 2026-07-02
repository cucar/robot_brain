# Stock & Time-Series Demos

These demos apply the brain to financial time series — synthetic cycles, real historical price/volume
data, and sequence memorization. Stock trading is **one application** of the architecture, not its
focus; for the vision and language demos see [mnist-demos.md](mnist-demos.md) and
[text-demos.md](text-demos.md).

The included `1D` timeframe data is ready to use — **no API key needed** for these demos. To download
fresh or different data, see [Downloading Fresh Stock Data](#downloading-fresh-stock-data) at the end.

---

## Single-Channel Synthetic Cycle

A single-stock variant of the cycle test: one channel, a repeating 12-frame price/volume pattern, 20
repeats. Isolated to a single channel so you can see the brain converge on optimal actions without
multi-channel reinforcement doing any of the work.

```bash
node apps/stocks/jobs/synthetic-extended-test.js --group-mode static --group-threshold 0.9
```

**Expected output:**
```
Overall Optimal Rate: 233/240 = 97.1%
```

With the right thresholds the brain converges to 97%+ optimal action decisions on a single-channel
cyclical pattern — confirming that hierarchy and action inference work without cross-channel consensus.

---

## Multi-Channel Synthetic Cycle

The brain learns to trade 3 stocks simultaneously (KGC, GLD, SPY), each as a separate channel. A
repeating 12-day price cycle is presented 20 times — the brain discovers cross-stock patterns and
converges on optimal buy/sell timing.

```bash
node apps/stocks/jobs/multi-channel-test.js --group-mode static --group-threshold 0.9
```

**Expected output:**
```
🎯 Overall Optimal Rate: 695/720 = 96.5%
```

The brain learns when to own vs. not own each stock based on upcoming price movements, achieving 96%+
optimal trade decisions across all three channels. This demonstrates how multiple input streams
converge to improve inference — one of the architecture's core strengths.

---

## Stock Trading

The brain learns to trade stocks from historical price and volume data. Each stock is a separate
channel — the brain discovers cross-stock patterns and makes buy/sell/hold decisions optimized by
reward feedback.

```bash
node apps/stocks/jobs/test.js --symbols SO,VALE,STLD,GOOGL,MU,PLTR,UUUU,PFE,CRM,HAL,AWR,GM,EQIX,RTX,KGC,ALB,AAPL,CVX,HD,WPM,BEP,AREC,JNJ,SLB,PLD,EXK,NVDA,CAT,WFC,RGLD,WEAT,OXY,CEG,LOW,PAAS,MP,LMT,GS,COST,AG,TECK,MRK,INTC,BIP,PSA,DVN,AVAV,PEP,CDE,TSM --context-length 3 --max-positions 3 --transaction-cost 0.02 --columns 20 --spatial
```

**Expected output:**
```
🎯 Final Training Results (1 episodes):
============================================================
📈 Overall Performance:
   Starting Capital: $15000.00
   Total Net Profit: $2562690.23
   Average per Episode: $2562690.23
   Average ROI: +17084.60%
   Average Per-Frame ROI: +0.095832%
   Average Sharpe Ratio: 0.42
   Total Transaction Cost: $621143.23 (0.02% per trade)
   Total Trades: 28811
   Average Trades per Episode: 28811.0

💰 Net Profit & ROI by Episode:
   Episode 1: $2562690.23 | ROI: +17084.60%, +0.095832%/frame, Sharpe: 0.42 (28811 trades)

📊 Base Level Accuracy by Episode:
   Episode 1: 11.04%
```

The brain achieves only ~11% base-level prediction accuracy because it's predicting price movement groups.
The **reward-weighted action selection** still turns a profit by learning which contexts produce better outcomes. 
With spatial co-activation enabled (`--spatial`), the brain trades direction accuracy for far more aggressive,
concentrated position sizing: it is wrong about direction more often than not, but sizes the contexts that pay off.

> **Reading this number honestly.** The *total* return over ~21 years of daily data (~5,370 frames) compounds to 
> **roughly +28%/year**, strong but far less dramatic than the raw percentage looks. 
> It is also measured on a *curated* 50-symbol universe of names that survived and performed well over the
> period (a survivorship bias), with low simulated friction and hyperparameters tuned in-sample. The
> base accuracy is **below 50%** and most of the edge comes from large, concentrated bets on a handful of volatile,
> low-priced names — restrict the universe to stable, higher-priced names (e.g. via `--min-price 10`) and the
> headline collapses to a far more sober ~+20%/year. The big number is a high-variance, in-sample artifact, not a
> reliable forward return.
>
> A more rigorous measurement uses a neutral, sector-diversified universe and a proper train/test split — train on
> the first ~80% of history, then evaluate on the **held-out final ~4.3 years** the brain never trained on.
> Out-of-sample performance peaks at **two training passes**, returning **~+20%/year at a real Sharpe of ~0.76**;
> additional passes push in-sample return higher while held-out return decays (textbook overfitting). The execution
> path has no look-ahead — the brain decides and trades at each day's close and is scored by the next day's move.

### Random Baseline Comparison

A natural worry is that the result might come from a favorable data window rather than learned trading. To rule that
out, the same job can be run with `--random-baseline`, which replaces the brain's action inference with a coin flip:
50% chance to be fully out, 50% chance to own one stock chosen uniformly at random. Everything else — the same data,
same portfolio sizing, same execution path — is unchanged.

```bash
node apps/stocks/jobs/test.js --random-baseline
```

`--random-baseline` skips encoding, reward collection, and `brain.processFrame()` entirely, so the run is purely the
trading harness driven by random signals — no shared work with the brain path.

**Example random-baseline run:**

![Random baseline run](../images/random_stocks.jpg)

Random coin-flips don't just underperform the brain — they *lose money*. A representative run ends at **−$1,839.36
(−12.26% ROI, Sharpe −0.19)** over the full 5,373 days: undirected exposure drifts up and down with whatever the
basket does, pays transaction costs on every flip, and finishes underwater. Swap that random decision source for the
brain — same data, same portfolio sizing, same execution path — and the result turns solidly positive. This is the
simplest sanity check that the brain is doing real work: identical harness, identical data, only the decision source
changes — and one bleeds capital while the other compounds.

---

## Action Learning in Low Accuracy

The brain learns the best actions to perform in each situation over repeated episodes, even when base prediction
accuracy is low. Note that the spatial processing here has an outsized impact.

```bash
node apps/stocks/jobs/test.js --symbols SO,VALE,STLD,GOOGL,MU,PLTR,UUUU,PFE,CRM,HAL --context-length 3 --columns 20 --no-summary --episodes 5 --spatial
```

**Expected output:**
```
💰 Net Profit & ROI by Episode:
   Episode 1: $830302.80 | ROI: +5535.35%, +0.075063%/frame, Sharpe: 0.22 (8479 trades)
   Episode 2: $44566910.12 | ROI: +297112.73%, +0.148948%/frame, Sharpe: 0.49 (8747 trades)
   Episode 3: $225986507.16 | ROI: +1506576.71%, +0.179208%/frame, Sharpe: 0.61 (8689 trades)
   Episode 4: $2109464653.55 | ROI: +14063097.69%, +0.220863%/frame, Sharpe: 0.75 (8682 trades)
   Episode 5: $29132233788.59 | ROI: +194214891.92%, +0.269846%/frame, Sharpe: 0.96 (8806 trades)

📊 Base Level Accuracy by Episode:
   Episode 1: 10.97%
   Episode 2: 8.63%
   Episode 3: 8.23%
   Episode 4: 8.29%
   Episode 5: 8.29%
```

The astronomical ROI is compounding over many episodes of in-sample data — a stress test of action learning, not a
forward return. The point is that even with single-digit base prediction accuracy, reward-weighted action selection
keeps improving the *actions* taken episode over episode.

---

## Stock Sequence Memorization

The brain memorizes a repeating stock price sequence across 5 episodes, reaching 60% prediction accuracy on real
market data. This demonstrates convergence on financial data — the same learning curve seen in text memorization.

```bash
node apps/stocks/jobs/test.js --no-summary --episodes 5 --symbols KGC,GOLD,SPY --context-length 3 --forget-rate 0.0005 --group-mode static --group-threshold 0.9
```

**Expected output:**
```
💰 Net Profit & ROI by Episode:
   Episode 1: $500326.56 | ROI: +3335.51%, +0.065846%/frame, Sharpe: 0.29 (5600 trades)
   Episode 2: $4824048976177885184.00 | ROI: +32160326507852568.00%, +0.623644%/frame, Sharpe: 3.79 (6038 trades)
   Episode 3: $3.986340595240165e+21 | ROI: +26575603968267767808.00%, +0.749517%/frame, Sharpe: 4.44 (6096 trades)
   Episode 4: $9.768420215418699e+22 | ROI: +651228014361246564352.00%, +0.809517%/frame, Sharpe: 4.78 (6101 trades)
   Episode 5: $6.235109589369968e+23 | ROI: +4.156739726246645e+21%, +0.844301%/frame, Sharpe: 4.96 (6026 trades)

📊 Base Level Accuracy by Episode:
   Episode 1: 51.61%
   Episode 2: 62.41%
   Episode 3: 65.13%
   Episode 4: 66.00%
   Episode 5: 66.59%
```

The brain goes from 50% accuracy (random) to 66% in 5 episodes on 3 stocks × ~5,300 frames of real market data. 
The low forget rate allows patterns to survive the full sequence. 
Short context (3 frames) is used because of computation limits.

---

## Downloading Fresh Stock Data

To download new data or different timeframes, you need a free [Alpaca](https://alpaca.markets) account:

1. Sign up at [alpaca.markets](https://alpaca.markets) (free paper trading account)
2. Get your API key and secret from the dashboard
3. Add them to `apps/stocks/.env` (copy `apps/stocks/.env.example`)
4. Run the downloader for the symbols and timeframe you want

The included `1D` data covers ~21 years and is enough to reproduce every demo above without an account.
