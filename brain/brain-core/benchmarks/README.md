# Benchmarks of the build

The benchmarks of the build at commit `61d9b31` on the branch `ucar` (the build to spec `df42b3b`, with the
speedups that keep every output the same). A port of this build's logic to another code structure is done when
every benchmark below gives the same results. Each benchmark's full output, timings stripped, is saved in
`output/`: `<name>.out` is what it printed to stdout and `<name>.err` what it printed to stderr.

## Running them

```
brain/brain-core/benchmarks/run.sh [mnist-data-dir]
```

The script builds the examples (`cargo build --release --examples`, from `brain/brain-core`), runs every benchmark
from `brain/brain-core`, strips the timings from what each printed and compares the rest, stdout and stderr, with
the saved output: `same` or `different` per benchmark, with its run time. It exits non-zero where any benchmark
differs. `--only <name>` runs one benchmark; `--record` saves each run's output as the expected one.

The timings stripped (the same rule for the saved and the new output): any number followed by `ms` or `ms/image`;
a number of seconds (`590s`, `| 56.9 s`) at the end of a line; a line that is a number of seconds alone. Nothing
else is changed.

The MNIST data are the four raw (gunzipped) idx files of the standard MNIST set. The script reads them from the
directory given, or from `apps/mnist/data` in this repository, where it downloads them from the mirror
`apps/mnist/jobs/download.js` uses (`https://storage.googleapis.com/cvdf-datasets/mnist/<file>.gz`) if they are
not there. Their MD5 sums:

| file | MD5 |
|---|---|
| `train-images-idx3-ubyte` | `6bbc9ace898e44ae57da46a324031adb` |
| `train-labels-idx1-ubyte` | `a25bea736e30d166cdddb491f175f624` |
| `t10k-images-idx3-ubyte` | `2646ac647ad5339dbf082846283269ea` |
| `t10k-labels-idx1-ubyte` | `27ae3e4e09519cfbb04c329615203637` |

The stock data are in `apps/stocks/data/1D` in this repository: `<symbol>.json` (daily bars; the `OpenPrice` of
the first 12 bars is read) for the lap benchmarks, `<symbol>.csv` (one row per day, the close first) for the real
stocks. The `channels` example reads that directory by default.

## Which benchmarks repeat

Every benchmark gives the same output, timings stripped, on every run: the outputs saved here were recorded once,
and a second run of the script from a clean checkout of the commit that holds them gave the same output for all
23. Nothing in the build reads the clock, a random number or the order of a hash map into a result; the one part
that runs in parallel (the neurons of a level, over the columns) gathers its results in a fixed order.

The run times are those of the recorded run, on a machine of 4 cores; they are for reference only and are not
compared.

## Summary

| benchmark | result | functions | last dictionary change | run time |
|---|---|---|---|---|
| alternation | 30/30 right | 1 | frame 8 | under 1 s |
| six-bar-lap | 165/178 right | 11 | frame 96 | under 1 s |
| kgc-gld | KGC 107/118, GLD 106/118 | 20 | frame 120 | under 1 s |
| two-six-bar-laps | A 177/179, B 166/179 | 13 | frame 100 | under 1 s |
| six-and-five-bar-laps | A 318/359, B 313/359 | 50 | frame 347 | under 1 s |
| three-stocks | KGC 1181/1199, GLD 1188/1199, SPY 1177/1199 | 43 | frame 1046 | 2.2 s |
| four-stocks | KGC 1181/1199, GLD 1190/1199, SPY 1171/1199, Q 1179/1199 | 50 | frame 1045 | 1.7 s |
| three-stocks-5-laps | KGC 15451/26864, GOLD 16364/26864, SPY 15528/26864 | 3389 | frame 26864 | 224.3 s |
| ten-stocks | SO 2768/5372, VALE 2730/5372, STLD 2662/5372, GOOGL 2639/5372, MU 2684/5372, PLTR 4678/5372, UUUU 3037/5372, PFE 2695/5372, CRM 2768/5372, HAL 2644/5372 | 2114 | frame 5373 | 43.4 s |
| tap | 37/37 right | 3 | not printed | under 1 s |
| text | 31/35 right | 4 | not printed | under 1 s |
| text-without-space | 20/35 right | 1 | not printed | under 1 s |
| hline | said 0 at 29/30 showings | 7 | showing 7 | under 1 s |
| vline | said 0 at 29/30 showings | 7 | showing 7 | under 1 s |
| filled-square | said 0 at 29/30 showings | 7 | showing 6 | under 1 s |
| hollow-square | said 0 at 29/30 showings | 5 | showing 5 | under 1 s |
| two-and-three | the 2 said 2 in 27/30 rounds, the 3 said 3 in 26/30; frozen: the 2 says 2, the 3 says 3 | 24 | round 24 | under 1 s |
| mnist-1x60 | 1/1 in the last episode; 59/60 over the run | 40 | episode 25 | 1.4 s |
| mnist-10x20 | 10/10 in the last episode; 187/200 over the run | 220 | episode 20 (still changing) | 5.4 s |
| mnist-10x300 | 10/10 in the last episode; 2987/3000 over the run | 229 | episode 298 | 126.5 s |
| mnist-255 | 96/255 | 641 | image 254 of 0..254 (still changing) | 6.9 s |
| mnist-100 | 19/100 | 289 | image 99 of 0..99 (still changing) | 2.3 s |
| mnist-1000 | 435/1000 | 2055 | image 999 of 0..999 (still changing) | 37.4 s |

## 1. The alternation

`alternation` — run time under 1 s

```
cargo run --release --example xy -- 33
```

Data: none: the letters x and y alternate for 33 frames (letter channel of 26 values, H 64).

| | |
|---|---|
| right (each prediction against the next letter) | 30/30 |
| wrong at frames | [] |
| neurons at the end | 27 |
| functions at the end | 1 |
| the dictionary last changed | frame 8 |
| the dictionary at the end | `f0 no origin [#24→#23[-1, 0, 0] #23→#24[-1, 0, 0]] cycle -> #26` |

## 2.1 The six-bar lap

`six-bar-lap` — run time under 1 s

```
cargo run --release --example stocks -- 30 256
```

Data: none: one stock, the bars +1% −2% −3% +2% +3% −1% repeated (rise fall fall rise rise fall), 30 laps = 180 bars, H 256.

| | |
|---|---|
| right, every inference read against the bar it was made for | 165/178 |
| right, the example's own count (the last frame's inference not graded) | 164/177 |
| lessons (frames where the machine said nothing) | 3 |
| first lap all right | 7 |
| neurons at the end | 15 |
| functions at the end | 11 |
| the dictionary last changed | frame 96 |

Per bar of the lap:

| bar | right | wrong | last 10 laps |
|---|---|---|---|
| 1 | 26 | 3 | 10/10 |
| 2 | 28 | 1 | 10/10 |
| 3 | 26 | 3 | 10/10 |
| 4 | 27 | 3 | 10/10 |
| 5 | 28 | 2 | 10/10 |
| 6 | 29 | 1 | 10/10 |

## 2.2 KGC and GLD

`kgc-gld` — run time under 1 s

```
cargo run --release --example channels -- two 10
```

Data: `apps/stocks/data/1D/KGC.json` and `GLD.json`: the first 12 daily `OpenPrice`s of each, their 12 moves (the last to the first) cycled 10 times = 120 frames; two channels.

| channel | right | of | every bar right from cycle |
|---|---|---|---|
| KGC | 107 | 118 | 5 |
| GLD | 106 | 118 | 5 |

| | |
|---|---|
| neurons at the end | 28 |
| functions at the end | 20 |
| the dictionary last changed | frame 120 |

## 2.3 Two stocks on six-bar laps

`two-six-bar-laps` — run time under 1 s

```
cargo run --release --example channels -- lap 30
```

Data: none: A +1 −2 −3 +2 +3 −1 and B +2 +1 −1 −3 +3 −2, each its own channel, 30 laps = 180 frames.

| channel | right | of | every bar right from cycle |
|---|---|---|---|
| A | 177 | 179 | 2 |
| B | 166 | 179 | 9 |

| | |
|---|---|
| neurons at the end | 21 |
| functions at the end | 13 |
| the dictionary last changed | frame 100 |

## 2.4 The six-bar and the five-bar lap

`six-and-five-bar-laps` — run time under 1 s

```
cargo run --release --example channels -- lap65 12
```

Data: none: A on its six-bar lap +1 −2 −3 +2 +3 −1, B on a five-bar lap +2 +1 −1 +3 −2; together they repeat every 30 frames; 12 cycles = 360 frames.

| channel | right | of | every bar right from cycle |
|---|---|---|---|
| A | 318 | 359 | 5 |
| B | 313 | 359 | 9 |

| | |
|---|---|
| neurons at the end | 58 |
| functions at the end | 50 |
| the dictionary last changed | frame 347 |

## 2.5 Three stocks

`three-stocks` — run time 2.2 s

```
cargo run --release --example channels -- three 100
```

Data: `apps/stocks/data/1D/KGC.json`, `GLD.json`, `SPY.json`: the first 12 daily `OpenPrice`s of each, cycled 100 times = 1,200 frames; three channels.

| channel | right | of | every bar right from cycle |
|---|---|---|---|
| KGC | 1181 | 1199 | 8 |
| GLD | 1188 | 1199 | 8 |
| SPY | 1177 | 1199 | 9 |

| | |
|---|---|
| neurons at the end | 55 |
| functions at the end | 43 |
| the dictionary last changed | frame 1046 |

## 2.6 Four stocks

`four-stocks` — run time 1.7 s

```
cargo run --release --example channels -- three4 100 --all4
```

Data: KGC, GLD and SPY as in 2.5, and Q, made up, moving one percent a bar U D D U U D U D D U D D; 12-bar lap cycled 100 times = 1,200 frames; four channels, every one graded (`--all4`).

| channel | right | of | every bar right from cycle |
|---|---|---|---|
| KGC | 1181 | 1199 | 7 |
| GLD | 1190 | 1199 | 5 |
| SPY | 1171 | 1199 | 9 |
| Q | 1179 | 1199 | 8 |

| | |
|---|---|
| neurons at the end | 66 |
| functions at the end | 50 |
| the dictionary last changed | frame 1045 |

## 3.1 Three stocks over 5 laps

`three-stocks-5-laps` — run time 224.3 s

```
cargo run --release --example channels -- seq 5
```

Data: `apps/stocks/data/1D/KGC.csv`, `GOLD.csv`, `SPY.csv`, 5,374 rows each (5,373 moves), their whole daily history as one lap, 5 laps = 26,865 frames; three channels.

| channel | right | of | every bar right from cycle |
|---|---|---|---|
| KGC | 15451 | 26864 | never |
| GOLD | 16364 | 26864 | never |
| SPY | 15528 | 26864 | never |

| | |
|---|---|
| neurons at the end | 3399 |
| functions at the end | 3389 |
| the dictionary last changed | frame 26864 |

Profit at the end of the run: $864439984317212655616.00 over 5197 trades (the last cycle's line).

Per lap:

| lap | neurons | functions | right per stock | profit (compounded, as printed) | trades |
|---|---|---|---|---|---|
| 1 | 443 | 431 | KGC 50.26%, GOLD 71.43%, SPY 50.37% | $88081.32 | 5854 |
| 2 | 586 | 574 | KGC 50.40%, GOLD 50.72%, SPY 51.80% | $8490500.58 | 6562 |
| 3 | 1344 | 1332 | KGC 54.49%, GOLD 53.49%, SPY 54.77% | $1126193540218.79 | 6546 |
| 4 | 2265 | 2254 | KGC 61.94%, GOLD 61.33%, SPY 61.59% | $238643236784832086016.00 | 6587 |
| 5 | 3399 | 3389 | KGC 70.48%, GOLD 67.62%, SPY 70.48% | $864439984317212655616.00 | 5197 |

The profit compounds over the laps and grows past any real figure; it is recorded as printed.

## 3.2 Ten stocks in one pass

`ten-stocks` — run time 43.4 s

```
cargo run --release --example channels -- seq 1 --symbols=SO,VALE,STLD,GOOGL,MU,PLTR,UUUU,PFE,CRM,HAL --max-positions=3 --cost=0.02 --every=100
```

Data: `apps/stocks/data/1D/<symbol>.csv` for SO, VALE, STLD, GOOGL, MU, PLTR, UUUU, PFE, CRM and HAL, 5,374 rows each (5,373 moves), one pass; at most three positions held at once, a cost of 0.02 a trade; a line every 100 frames.

| channel | right | of | every bar right from cycle |
|---|---|---|---|
| SO | 2768 | 5372 | never |
| VALE | 2730 | 5372 | never |
| STLD | 2662 | 5372 | never |
| GOOGL | 2639 | 5372 | never |
| MU | 2684 | 5372 | never |
| PLTR | 4678 | 5372 | never |
| UUUU | 3037 | 5372 | never |
| PFE | 2695 | 5372 | never |
| CRM | 2768 | 5372 | never |
| HAL | 2644 | 5372 | never |

| | |
|---|---|
| neurons at the end | 2154 |
| functions at the end | 2114 |
| the dictionary last changed | frame 5373 |

Profit at the end of the run: $410959.57 over 25603 trades (the last cycle's line).

Average of the ten percents right: 54.55%. Cycles closed: 30 (cycles at the end 30 of 2114 functions).

Every 100 frames:

| frame | neurons | functions | right, last 100 | right so far | net profit | trades |
|---|---|---|---|---|---|---|
| 100 | 194 | 154 | 60.91% | 60.91% | $2496.40 | 419 |
| 200 | 298 | 258 | 61.50% | 61.21% | $5534.57 | 875 |
| 300 | 369 | 329 | 58.20% | 60.20% | $11151.15 | 1339 |
| 400 | 438 | 398 | 60.70% | 60.33% | $13447.42 | 1802 |
| 500 | 501 | 461 | 61.40% | 60.54% | $14940.71 | 2262 |
| 600 | 557 | 517 | 59.40% | 60.35% | $18614.83 | 2736 |
| 700 | 600 | 560 | 53.10% | 59.31% | $29251.72 | 3232 |
| 800 | 644 | 604 | 56.50% | 58.96% | $15771.35 | 3696 |
| 900 | 694 | 654 | 55.90% | 58.62% | $17328.66 | 4138 |
| 1000 | 731 | 691 | 55.30% | 58.29% | $30187.39 | 4592 |
| 1100 | 786 | 746 | 52.50% | 57.76% | $21809.43 | 5058 |
| 1200 | 825 | 785 | 56.30% | 57.64% | $40837.09 | 5521 |
| 1300 | 863 | 823 | 58.10% | 57.68% | $57300.06 | 5988 |
| 1400 | 891 | 851 | 55.90% | 57.55% | $60121.49 | 6460 |
| 1500 | 931 | 891 | 55.70% | 57.42% | $72332.51 | 6908 |
| 1600 | 959 | 919 | 57.90% | 57.45% | $67450.30 | 7377 |
| 1700 | 969 | 929 | 53.10% | 57.20% | $44900.54 | 7848 |
| 1800 | 996 | 956 | 55.80% | 57.12% | $79691.36 | 8312 |
| 1900 | 1023 | 983 | 55.20% | 57.02% | $70001.56 | 8789 |
| 2000 | 1051 | 1011 | 53.30% | 56.83% | $79047.42 | 9273 |
| 2100 | 1084 | 1044 | 54.90% | 56.74% | $86548.09 | 9732 |
| 2200 | 1107 | 1067 | 57.70% | 56.78% | $105866.66 | 10192 |
| 2300 | 1146 | 1106 | 52.30% | 56.59% | $103712.68 | 10672 |
| 2400 | 1167 | 1127 | 56.10% | 56.57% | $104362.61 | 11138 |
| 2500 | 1186 | 1146 | 56.50% | 56.57% | $94167.18 | 11612 |
| 2600 | 1219 | 1179 | 55.80% | 56.54% | $106833.69 | 12094 |
| 2700 | 1245 | 1205 | 55.70% | 56.51% | $76786.20 | 12561 |
| 2800 | 1265 | 1225 | 52.90% | 56.38% | $79423.23 | 13031 |
| 2900 | 1302 | 1262 | 59.60% | 56.49% | $116472.27 | 13508 |
| 3000 | 1315 | 1275 | 54.00% | 56.41% | $122945.92 | 13994 |
| 3100 | 1340 | 1300 | 54.70% | 56.35% | $128248.92 | 14484 |
| 3200 | 1372 | 1332 | 53.70% | 56.27% | $134348.11 | 14970 |
| 3300 | 1398 | 1358 | 54.70% | 56.22% | $140462.39 | 15467 |
| 3400 | 1421 | 1381 | 53.00% | 56.13% | $157470.77 | 15951 |
| 3500 | 1445 | 1405 | 55.00% | 56.09% | $152294.66 | 16436 |
| 3600 | 1470 | 1430 | 54.60% | 56.05% | $134797.43 | 16932 |
| 3700 | 1509 | 1469 | 54.60% | 56.01% | $113349.97 | 17414 |
| 3800 | 1550 | 1510 | 57.00% | 56.04% | $128247.84 | 17882 |
| 3900 | 1582 | 1542 | 55.20% | 56.02% | $114563.81 | 18374 |
| 4000 | 1620 | 1580 | 52.90% | 55.94% | $108318.06 | 18859 |
| 4100 | 1663 | 1623 | 48.30% | 55.75% | $152391.72 | 19364 |
| 4200 | 1705 | 1665 | 50.60% | 55.63% | $156724.61 | 19875 |
| 4300 | 1748 | 1708 | 51.30% | 55.53% | $175182.07 | 20376 |
| 4400 | 1779 | 1739 | 50.60% | 55.42% | $141238.90 | 20881 |
| 4500 | 1816 | 1776 | 52.60% | 55.35% | $154787.52 | 21379 |
| 4600 | 1851 | 1811 | 50.90% | 55.26% | $188171.97 | 21839 |
| 4700 | 1874 | 1834 | 49.60% | 55.14% | $226119.72 | 22312 |
| 4800 | 1920 | 1880 | 52.00% | 55.07% | $191548.00 | 22783 |
| 4900 | 1958 | 1918 | 52.00% | 55.01% | $205541.61 | 23255 |
| 5000 | 2005 | 1965 | 50.90% | 54.93% | $244766.48 | 23747 |
| 5100 | 2044 | 2004 | 51.90% | 54.87% | $179067.55 | 24226 |
| 5200 | 2078 | 2038 | 48.40% | 54.74% | $214958.96 | 24727 |
| 5300 | 2123 | 2083 | 50.10% | 54.66% | $361261.30 | 25242 |

## 4.1 Tap

`tap` — run time under 1 s

```
cargo run --release --example tap -- 6 8
```

Data: none: the cues go, go, go, stop; six teaching stretches, silence of 8 frames between; then the test.

| | |
|---|---|
| right in the test | 37/37 |
| functions at the end | 3 |
| neurons | not printed as a count; the dump at the end lists them |
| the dictionary last changed | not printed |

## 4.2 Text

`text` — run time under 1 s

```
cargo run --release --example text -- "the the the the the the the the the " 1 64
```

Data: none: the text `the the the the the the the the the ` streamed letter by letter through one letter channel, R 1, H 64.

| | |
|---|---|
| right (each letter predicted) | 31/35 |
| functions (patterns) at the end | 4 |
| neurons | 260 alive of 262 |
| the dictionary last changed | not printed |

## 4.3 Text without the recurring space

`text-without-space` — run time under 1 s

```
cargo run --release --example text -- "theathebthecthedthefthegtheithejthek" 1 64
```

Data: none: the text `theathebthecthedthefthegtheithejthek` streamed letter by letter through one letter channel, R 1, H 64.

| | |
|---|---|
| right (each letter predicted) | 20/35 |
| functions (patterns) at the end | 1 |
| neurons | 257 alive of 258 |
| the dictionary last changed | not printed |

## 5.1 The horizontal line

`hline` — run time under 1 s

```
cargo run --release --example shapes -- hline 30
```

Data: none: a white horizontal line of 5 pixels in the middle row of a black 7x5 image, shown 30 times to one machine, each showing followed by the digit 0 with a reward of +1 and a reset of the context.

| | |
|---|---|
| said 0 | 29 of 30 showings (the first says nothing) |
| functions at the end | 7 |
| neurons | not printed |
| the dictionary last changed | showing 7 |
| on the apex at the end | 1 call |

## 5.2 The vertical line

`vline` — run time under 1 s

```
cargo run --release --example shapes -- vline 30
```

Data: none: a white vertical line of 5 pixels in the middle column of a black 5x7 image, shown 30 times to one machine, each showing followed by the digit 0 with a reward of +1 and a reset of the context.

| | |
|---|---|
| said 0 | 29 of 30 showings (the first says nothing) |
| functions at the end | 7 |
| neurons | not printed |
| the dictionary last changed | showing 7 |
| on the apex at the end | 1 call |

## 5.3 The filled square

`filled-square` — run time under 1 s

```
cargo run --release --example shapes -- square 30
```

Data: none: a 5x5 image, a black border around a white 3x3 middle, shown 30 times to one machine, each showing followed by the digit 0 with a reward of +1 and a reset of the context.

| | |
|---|---|
| said 0 | 29 of 30 showings (the first says nothing) |
| functions at the end | 7 |
| neurons | not printed |
| the dictionary last changed | showing 6 |
| on the apex at the end | 1 call |

## 5.4 The hollow square

`hollow-square` — run time under 1 s

```
cargo run --release --example shapes -- hollow 30
```

Data: none: a 5x5 black border whose 3x3 middle reports nothing, shown 30 times to one machine, each showing followed by the digit 0 with a reward of +1 and a reset of the context.

| | |
|---|---|
| said 0 | 29 of 30 showings (the first says nothing) |
| functions at the end | 5 |
| neurons | not printed |
| the dictionary last changed | showing 5 |
| on the apex at the end | 1 call |

## 5.5 The 2 and the 3 taught in turn

`two-and-three` — run time under 1 s

```
cargo run --release --example shapes -- teach23 30 0 -
```

Data: none: the images of a 2 and a 3, taught in turn with their digit for 30 rounds, then each shown frozen without it.

| | |
|---|---|
| the 2 said 2 | 27 of 30 rounds |
| the 3 said 3 | 26 of 30 rounds |
| frozen, the 2 without its digit | says 2 |
| frozen, the 3 without its digit | says 3 |
| functions at the end | 24 |
| edges at the end | 38 |
| the dictionary last changed | round 24 |

## 6.1 One image over 60 episodes

`mnist-1x60` — run time 1.4 s

```
cargo run --release --example mnist -- <mnist-data-dir> 1 0 2048 14 all 128 0 all 60
```

Data: MNIST: the first 1 training image of `train-images-idx3-ubyte` in the file's order, 14x14 (each 2x2 block of the 28x28 image averaged), ink where the block's mean reaches 128 and blank elsewhere; H 2048; every training image a lesson (the world runs the image's digit, teach = all); no test images. Shown over 60 episodes.

| | |
|---|---|
| right in the last episode | 1/1 |
| right over the run | 59 of 60 |
| functions at the end | 40 |
| neurons at the end | 46 |
| the dictionary last changed | episode 25 |

Episodes (all are in the saved output):

| episode | dictionary | functions | right | calls on the apex per image |
|---|---|---|---|---|
| 1 | changed | 6 | 0/1 | 28.0 |
| 5 | changed | 23 | 1/1 | 9.0 |
| 10 | changed | 30 | 1/1 | 5.0 |
| 20 | changed | 39 | 1/1 | 2.0 |
| 50 | same | 40 | 1/1 | 1.0 |
| 60 | same | 40 | 1/1 | 1.0 |

## 6.2 10 images over 20 episodes

`mnist-10x20` — run time 5.4 s

```
cargo run --release --example mnist -- <mnist-data-dir> 10 0 2048 14 all 128 0 all 20
```

Data: MNIST: the first 10 training images of `train-images-idx3-ubyte` in the file's order, 14x14 (each 2x2 block of the 28x28 image averaged), ink where the block's mean reaches 128 and blank elsewhere; H 2048; every training image a lesson (the world runs the image's digit, teach = all); no test images. Shown over 20 episodes.

| | |
|---|---|
| right in the last episode | 10/10 |
| right over the run | 187 of 200 |
| functions at the end | 220 |
| neurons at the end | 176 |
| the dictionary last changed | episode 20 (still changing) |

Episodes (all are in the saved output):

| episode | dictionary | functions | right | calls on the apex per image |
|---|---|---|---|---|
| 1 | changed | 36 | 1/10 | 19.4 |
| 5 | changed | 136 | 10/10 | 6.2 |
| 10 | changed | 189 | 10/10 | 3.2 |
| 20 | changed | 220 | 10/10 | 1.1 |

## 6.3 10 images over 300 episodes

`mnist-10x300` — run time 126.5 s

```
cargo run --release --example mnist -- <mnist-data-dir> 10 0 2048 14 all 128 0 all 300
```

Data: MNIST: the first 10 training images of `train-images-idx3-ubyte` in the file's order, 14x14 (each 2x2 block of the 28x28 image averaged), ink where the block's mean reaches 128 and blank elsewhere; H 2048; every training image a lesson (the world runs the image's digit, teach = all); no test images. Shown over 300 episodes.

| | |
|---|---|
| right in the last episode | 10/10 |
| right over the run | 2987 of 3000 |
| functions at the end | 229 |
| neurons at the end | 181 |
| the dictionary last changed | episode 298 |

Episodes (all are in the saved output):

| episode | dictionary | functions | right | calls on the apex per image |
|---|---|---|---|---|
| 1 | changed | 36 | 1/10 | 19.4 |
| 5 | changed | 136 | 10/10 | 6.2 |
| 10 | changed | 189 | 10/10 | 3.2 |
| 20 | changed | 220 | 10/10 | 1.1 |
| 50 | same | 224 | 10/10 | 1.0 |
| 100 | same | 226 | 10/10 | 1.1 |
| 150 | same | 224 | 10/10 | 1.0 |
| 200 | same | 226 | 10/10 | 1.0 |
| 250 | same | 228 | 10/10 | 1.1 |
| 300 | same | 229 | 10/10 | 1.0 |

## 6.4 Images 1 to 255

`mnist-255` — run time 6.9 s

```
cargo run --release --example mnist -- <mnist-data-dir> 255 0 2048 14 all 128 0 all 1
```

Data: MNIST: the first 255 training images of `train-images-idx3-ubyte` in the file's order, 14x14 (each 2x2 block of the 28x28 image averaged), ink where the block's mean reaches 128 and blank elsewhere; H 2048; every training image a lesson (the world runs the image's digit, teach = all); no test images.

| | |
|---|---|
| right | 96 of 255 |
| functions at the end | 641 |
| neurons at the end | 626 |
| calls on the apex per image, over the run | 14.7 |
| the dictionary last changed | at image 254 (images 0 to 254): still changing |

Every 100 images (from stderr; calls on the apex per image is the mean so far):

| images | spoke, last 100 | right, last 100 | functions added | retired | neurons | functions | calls on the apex per image |
|---|---|---|---|---|---|---|---|
| 100 | 99 | 19 | 309 | 20 | 297 | 289 | 16.5 |
| 200 | 100 | 47 | 264 | 22 | 526 | 531 | 15.0 |

## 6.5 100 images in one pass

`mnist-100` — run time 2.3 s

```
cargo run --release --example mnist -- <mnist-data-dir> 100 0 2048 14 all 128 0 all 1
```

Data: MNIST: the first 100 training images of `train-images-idx3-ubyte` in the file's order, 14x14 (each 2x2 block of the 28x28 image averaged), ink where the block's mean reaches 128 and blank elsewhere; H 2048; every training image a lesson (the world runs the image's digit, teach = all); no test images.

| | |
|---|---|
| right | 19 of 100 |
| functions at the end | 289 |
| neurons at the end | 297 |
| calls on the apex per image, over the run | 16.5 |
| the dictionary last changed | at image 99 (images 0 to 99): still changing |

Every 100 images (from stderr; calls on the apex per image is the mean so far):

| images | spoke, last 100 | right, last 100 | functions added | retired | neurons | functions | calls on the apex per image |
|---|---|---|---|---|---|---|---|
| 100 | 99 | 19 | 309 | 20 | 297 | 289 | 16.5 |

## 6.6 1,000 images in one pass

`mnist-1000` — run time 37.4 s

```
cargo run --release --example mnist -- <mnist-data-dir> 1000 0 2048 14 all 128 0 all 1
```

Data: MNIST: the first 1000 training images of `train-images-idx3-ubyte` in the file's order, 14x14 (each 2x2 block of the 28x28 image averaged), ink where the block's mean reaches 128 and blank elsewhere; H 2048; every training image a lesson (the world runs the image's digit, teach = all); no test images.

| | |
|---|---|
| right | 435 of 1000 |
| functions at the end | 2055 |
| neurons at the end | 1903 |
| calls on the apex per image, over the run | 12.4 |
| the dictionary last changed | at image 999 (images 0 to 999): still changing |

Every 100 images (from stderr; calls on the apex per image is the mean so far):

| images | spoke, last 100 | right, last 100 | functions added | retired | neurons | functions | calls on the apex per image |
|---|---|---|---|---|---|---|---|
| 100 | 99 | 19 | 309 | 20 | 297 | 289 | 16.5 |
| 200 | 100 | 47 | 264 | 22 | 526 | 531 | 15.0 |
| 300 | 100 | 54 | 251 | 44 | 713 | 738 | 14.3 |
| 400 | 100 | 50 | 231 | 53 | 873 | 916 | 13.7 |
| 500 | 100 | 45 | 247 | 47 | 1051 | 1116 | 13.4 |
| 600 | 100 | 41 | 256 | 48 | 1234 | 1323 | 13.1 |
| 700 | 100 | 42 | 226 | 47 | 1401 | 1502 | 12.9 |
| 800 | 100 | 49 | 234 | 43 | 1576 | 1691 | 12.7 |
| 900 | 100 | 42 | 229 | 57 | 1733 | 1863 | 12.6 |
| 1000 | 100 | 46 | 240 | 48 | 1903 | 2055 | 12.4 |
