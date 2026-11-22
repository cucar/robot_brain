# Stock & Time-Series Demos

These demos apply the brain to financial time series: laps that repeat, real daily history with a trader, and
sequence memorization. Stock trading is **one application** of the architecture, not its focus; for the other demos
see [mnist-demos.md](mnist-demos.md), [text-demos.md](text-demos.md), [shape-demos.md](shape-demos.md) and
[tap-demos.md](tap-demos.md).

Each stock is one channel: the price's move from one bar to the next, up or down, is the event, and own or out is the
action. The reward of a bar is the move, signed by the action that stood over it, scoped to that stock's channel and
that one frame. Where the brain says nothing for a stock, the world runs the move itself as a lesson. With `--volume`,
each stock also gets a channel of its own for its volume's move; it is off by default.

The daily histories fill the days before a stock was listed, and the days it did not trade, with its price held and a
volume of 0 (PLTR before its listing in September 2020, most of GOLD's early years, some of UUUU's). No trading
happened on such a day, so the jobs skip it: the stock reports nothing that frame, runs nothing and earns nothing, and
a prediction made for it is not graded.

The included `1D` timeframe data is ready to use — **no API key needed** for these demos. To download fresh or
different data, see [Downloading Fresh Stock Data](#downloading-fresh-stock-data) at the end.

---

## One stock on a lap

KGC alone: the first 12 daily opens of its history, their 12 moves (the last to the first) shown 20 times over. With
no other channel, the brain has nothing to lean on but the one stock's lap.

```bash
node apps/stocks/jobs/synthetic-extended-test.js
```

**Expected output:**
```
CYCLE 20 | neurons 11 | functions 7 | highest call level in the cycle 3 | the dictionary last changed at frame 73 | right KGC 100.00%
inference KGC: right 225 of 238 | every bar right from cycle 8
```

225 of 238 bars right (94.5%), and every bar right from the eighth lap on.

## The six-bar lap

One stock on the bars +1% −2% −3% +2% +3% −1%, repeated for 30 laps.

```bash
node apps/stocks/jobs/synthetic-cycle-test.js 30
```

| | |
|---|---|
| right, every inference against the bar it was made for | 165/178 |
| lessons (frames where the brain said nothing) | 3 |
| first lap all right | 7 |
| functions at the end | 11 |
| the dictionary last changed | frame 96 |

Every bar of the lap is right over the last 10 laps.

## Three stocks on a lap

KGC, GLD and SPY, each the first 12 daily opens of its history, shown 20 times over; one channel each.

```bash
node apps/stocks/jobs/multi-channel-test.js
```

**Expected output:**
```
CYCLE 20 | neurons 58 | functions 46 | highest call level in the cycle 7 | the dictionary last changed at frame 235 | right KGC 100.00%, GLD 100.00%, SPY 100.00%
inference KGC: right 221 of 239 | every bar right from cycle 8
inference GLD: right 228 of 239 | every bar right from cycle 8
inference SPY: right 217 of 239 | every bar right from cycle 9
```

666 of 717 bars right (92.9%) over the run, and every bar of every stock right from the ninth lap on.

### More laps

`multi-channel-test.js --mode <mode> --laps <n>` runs the other laps. Over 100 laps of the three stocks, and with a
made-up fourth stock Q moving one percent a bar (U D D U U D U D D U D D), the brain gets 98–99% of the bars right;
over the shorter runs, 87–99%:

| command | right | of | every bar right from lap | functions |
|---|---|---|---|---|
| `--mode two --laps 10` (KGC, GLD) | KGC 107, GLD 106 | 118 each | 5, 5 | 20 |
| `--mode lap --laps 30` (A +1 −2 −3 +2 +3 −1, B +2 +1 −1 −3 +3 −2) | A 177, B 166 | 179 each | 2, 9 | 13 |
| `--mode lap65 --laps 12` (A on six bars, B on five: they repeat together every 30) | A 318, B 311 | 359 each | 5, 11 | 50 |
| `--mode three --laps 100` | KGC 1181, GLD 1188, SPY 1177 | 1199 each | 8, 8, 9 | 43 |
| `--mode three4 --laps 100 --all4` | KGC 1181, GLD 1190, SPY 1171, Q 1179 | 1199 each | 7, 5, 9, 8 | 50 |

---

## Stock Trading

Ten stocks over their whole daily history (5,373 bars, January 2005 to May 2026), one channel each, with a trader. At
each bar the trader brings its holdings in line with what the brain last said: each stock is owned or out, at most
three owned at once, the value split evenly among them, a cost of 0.02% on every trade. It starts with $15,000.

```bash
node apps/stocks/jobs/test.js --symbols SO,VALE,STLD,GOOGL,MU,PLTR,UUUU,PFE,CRM,HAL --max-positions 3 --transaction-cost 0.02
```

**Expected output** (the end of the lap):
```
TRADER lap 1 | net profit $36844.68 | ROI +245.63% | +0.023085%/frame | Sharpe 0.02 | transaction cost $25923.67 | trades 25028
CYCLE 1 | neurons 1829 | functions 1789 | highest call level in the cycle 4 | the dictionary last changed at frame 5373 | right SO 50.63%, VALE 50.78%, STLD 50.60%, GOOGL 50.97%, MU 50.04%, PLTR 50.00%, UUUU 50.12%, PFE 51.45%, CRM 49.24%, HAL 50.99% | net profit $36844.68, ROI 245.63% (25028 trades)
```

**In one pass, the brain does not predict the next day's move.** Every stock is right on about half of its real bars
(49–51%), and the trader's Sharpe ratio is 0.02: $15,000 becomes $51,845, about 6.0% a year, which is what holding a
basket of risky stocks pays, with no skill in it. The run takes about 30 seconds.

The ride is rough. The portfolio holds three stocks on 82% of the bars, nearly always fully invested, with no cash
cushion:

| | |
|---|---|
| drawdowns deeper than 10% / 20% / 30% / 50% | 12 / 6 / 2 / 1 |
| the deepest | −77.2%, from June 2008 ($27,382) to March 2020 ($6,256); back above the 2008 peak only in December 2023 |
| daily volatility | 2.3% (36.9% a year) |
| days moving more than 5% / 10% | 202 / 31 |
| worst and best day | −31.6% (15 July 2019), +23.5% (5 January 2009) |
| losing years | 2005, 2007, 2008, 2011, 2015, 2018, 2019 (the worst 2008, −48.3%) |

What the brain does with these stocks is memorize their history, lap after lap: the next demos after the random
baseline.

### Random Baseline Comparison

To see what the trader earns without the brain, the same job runs with `--random-baseline`: the brain is left out, and
at each bar a coin says all out half the time and, the other half, own one of the stocks picked at random. The trader
sizes and trades on that exactly as it does on the brain's actions. The coin is seeded (`--seed`, 1 by default), so a
run repeats; different seeds are different draws.

```bash
node apps/stocks/jobs/test.js --symbols SO,VALE,STLD,GOOGL,MU,PLTR,UUUU,PFE,CRM,HAL --random-baseline --seed 1
```

Over seeds 1 to 10, against the brain's single pass over the same ten stocks with the same settings:

| settings | random: profit, seeds 1–10 | random: Sharpe | the brain, first pass | beats |
|---|---|---|---|---|
| one stock at a time, no cost | $7,819 to $2,625,508 (middle about $68,000) | −0.08 to 0.58 | $621,303, Sharpe 0.22 | 9 of 10 seeds |
| three stocks, 0.02% cost | −$6,883 to $915,496 (middle about $15,000) | −0.22 to 0.44 | $36,845, Sharpe 0.02 | 7 of 10 seeds |

The coin owns at most one stock in either row; the settings are the brain's, and the cost is the coin's too. On its
first pass the brain earns more than most coins but not more than a lucky one: with daily moves right half the
time, its single-pass profit is within the range chance gives. Each random run takes under a second.

## Action Learning in Low Accuracy

The same ten stocks over five laps of their history, one stock owned at a time, no transaction cost. The brain keeps
what it learned from lap to lap; the trader starts each lap from $15,000.

```bash
node apps/stocks/jobs/test.js --symbols SO,VALE,STLD,GOOGL,MU,PLTR,UUUU,PFE,CRM,HAL --episodes 5
```

| lap | right, average of the ten | Sharpe | per-frame return | trades |
|---|---|---|---|---|
| 1 | 50.48% | 0.22 | +0.070% | 9,062 |
| 2 | 53.42% | 2.01 | +0.455% | 9,184 |
| 3 | 58.46% | 3.79 | +0.839% | 9,185 |
| 4 | 67.05% | 6.18 | +1.551% | 9,168 |
| 5 | 77.61% | 8.87 | +2.289% | 9,155 |

All ten stocks climb together: in lap 5 every one is right on 76–79% of its bars. The trader's return per bar and its
Sharpe ratio climb with them, and ahead of the share right: at 53.42% right in lap 2 the Sharpe ratio is already 2.01.
The first lap's
profit is $621,303 (ROI +4,142%); from there the profit compounds within each lap past any real figure, so the
per-frame return and the Sharpe ratio are the numbers to read. This is memorization of the history the brain trades
on, not a forward return. The run takes about 7 minutes.

## Stock Sequence Memorization

KGC, GOLD and SPY over their whole history (5,373 moves each), five laps, one stock owned at a time, no cost.

```bash
node apps/stocks/jobs/test.js --symbols KGC,GOLD,SPY --episodes 5
```

| lap | right KGC | right GOLD | right SPY | Sharpe | functions |
|---|---|---|---|---|---|
| 1 | 50.43% | 49.51% | 50.43% | −0.08 | 484 |
| 2 | 53.36% | 54.15% | 53.88% | 1.22 | 807 |
| 3 | 57.17% | 61.77% | 57.66% | 2.76 | 1,270 |
| 4 | 62.35% | 70.33% | 62.76% | 4.33 | 2,011 |
| 5 | 67.06% | 80.05% | 68.43% | 5.52 | 2,265 |

From a coin flip in the first lap to 67–80% in the fifth, on real market data. GOLD is graded only on its real bars, 15,232 over the five laps, against 26,864 for KGC and SPY. The run
takes about 2½ minutes.

---

## Downloading Fresh Stock Data

To download new data or different timeframes, you need a free [Alpaca](https://alpaca.markets) account:

1. Sign up at [alpaca.markets](https://alpaca.markets) (free paper trading account)
2. Get your API key and secret from the dashboard
3. Add them to `apps/stocks/.env` (copy `apps/stocks/.env.example`)
4. Run the downloader for the symbols and timeframe you want

The included `1D` data covers ~21 years and is enough to reproduce every demo above without an account.

The six-bar lap, the laps of the table above, the trading run and the sequence memorization are also benchmarks (see
[Benchmarks](../README.md#benchmarks)).
