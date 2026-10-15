# UCAR implementation: first pass and results

A from-scratch implementation of [algorithm.md](algorithm.md) as a new Rust crate, with a Node
binding and one job per application. The old engine (`brain-core`, `brain-napi`) is untouched; the
new one lives beside it so both can run.

| Piece | Where | What |
|---|---|---|
| core | `brain/ucar-core/` | the machine: `types.rs` (D1, D3, D35, D50, D51), `offsets.rs` (D6, D13's offset alphabets), `neuron.rs` (§4, §5), `machine.rs` (§6, §7, §8) |
| tests | `brain/ucar-core/tests/machine.rs`, `offsets.rs` | D6 rounding; a repeated pair gets a function and a child; varied predecessors form a class; prediction climbs over passes; a closed loop acts on the estimate; weak activations are connected to by nothing |
| examples | `brain/ucar-core/examples/` | `text` (letter stream with per-phase timings), `world` (synthetic closed loop with rewards), `dump` (print tables) |
| binding | `brain/ucar-napi/`, `brain/src/ucar.js`, `brain/build-ucar.sh` | the `Ucar` class: `declareAxis`, `declareChannel`, `processFrame(Int32Array, rewards)`, `resetContext`, `stats`, `levelCounts`, `dumpTables` |
| apps | `apps/text/jobs/ucar.js`, `apps/stocks/jobs/ucar.js`, `apps/mnist/jobs/ucar.js` | standalone jobs (import the binding by path; no workspace install needed) |

Build and run:

```
brain/build-ucar.sh                                   # cargo build --release -p ucar-napi, copies ucar-napi.node
cd brain/ucar-core && cargo test --release             # 7 tests
node apps/text/jobs/ucar.js --file test.txt --episodes 5 --h 256
node apps/stocks/jobs/ucar.js --symbols AAPL,MSFT,NVDA --episodes 2 --h 256 --max-frames 2500 --holdout 500
node apps/mnist/jobs/ucar.js --image-size 14 --per-class 100 --teach 300 --test 500 --h 2048
```

Every job takes `--dump` to print tables at the end. The first-pass tables below that compare "claim" and "no-claim" were measured under a regime that has since been removed (see "Two regimes"); the second pass supersedes them.

## What is implemented

**World and machine (§2, §3).** Activation dimensions (time plus up to two spatial axes, each with its
radius), channels with one event and/or one action dimension laid out over a subset of the axes, base
neurons minted in declaration order, frames of inputs with positions, rewards scoped to channels and a
span (D35), the neighborhood policy as two 3×3 tables (D50) and the action policy (D51). Offsets are
rounded to one significant digit in base R+1 (D6); prices are D13's fields in bits with `n` the live
neuron count and `B` the declared alphabet.

**Neuron (§4, §5).** The history ring of H activations, each with its neighborhood, own cost, cover and
owners (D18, D17, D19). Patterns: functions naming neighbors and variables; classes of one position
with members and value neurons (D41); parameters over two offsets that agree, with members and value
neurons (D44). The greedy cover by ratio with first-claimant owners (D28); D22's prices with failed
neighbors; the collapse for functions (every thing taken when what it covers beats its failures and its
entry, iterated on |p|), for classes (members in order of count while each pays its entry and the
widening; members never together in a row) and for parameters (members are what stood at all positions);
re-centering on the rows a pattern covers (D29); margins and retirement (D30, R18, R41); the greedy
pick, variables first then functions over rows read in things, each seeded on a relation counted at
least twice and priced by its margin (D33, R14, R15); merging of identical lines (D43); bids of each
fresh row's cover, functions sharing a variable bundled (D31, R21).

**Contraction (§6).** The level stack: activate children bought for the level, call every neuron with
an activation there, elect, create children, up to the last level with anything (R26). The election is
D28 over the board with R22's coverage and price, R24's tie order, coverage acquired late and never
revoked (R27), children one level above the highest activation covered, reuse of a child between
identical accepted bids (R43), value neurons per value (R44), wiring, spans and reaches recomputed when
lines move (D4), and the death ledger with deletion once the last window closes (R38).

**Connect and infer (§7).** Every strong open activation connects at its age to every strong apex
activation within reach the future table admits, or to its coverer in the frame it is covered (D10, R31);
connections hold strength and earnings; rewards land on the frame's connections with linearly falling
shares over the span and only on base actions of the named channels when scoped (R33); a negative
estimate wires the next untried base action, in each dimension the target acted in (R37); `infer` at age
0 snapshots the neuron's connections as standing inferences (R36, R45).

**Selection (§8).** Standing inferences are placed at their completion and expanded through dictionary
lines to base symbols with offsets composed (R28), bound values from the value neurons beside a call
or the most frequent member otherwise; the electorate is every uncovered open activation plus, where a
coverer has nothing for the frame ahead, what it covers (R46, R48); one winner per dimension and
position, actions by share-weighted estimate, events by share (R47), with the least-bad or nothing
policy (D51); winners fire weakly in the frame ahead, are replaced by the report where it names the
dimension, last one frame, write nothing and are connected to by nothing (D40, R40).

## What is not implemented or deviates

- **Match (R49)** is not implemented: a standing inference is proposed wherever its placements land,
  never struck for failing to match the past. Plans therefore are not worn down.
- **Classes have one position; parameters have exactly two.** The position tests of D27 (an offset
  joining or leaving a variable) are not run, and a class stores no combinations beyond one position.
- **Every row is re-covered on every call** rather than only rows whose residual changed (§5.2). The
  stricter rule is available (`UCAR_DIRTY_ONLY=1`) and is 2-4× faster on text, but it left stale covers
  that cost 10 points in the closed-loop world. The staleness §2 of the evaluation document anticipates
  is real.
- **A candidate function is priced over the rows that hold its seed**, not over the whole history
  (R15 says every activation where it pays). Pricing over all rows (`UCAR_ALL_ROWS=1`) scored the same
  on the world and costs about 3× in the pick.
- **Re-centering runs only for patterns whose covered rows changed** (a fresh row, an evicted row), as
  D29 says, but a pattern no fresh row reaches is never revisited.
- **The greedy pick tries at most 12 seeds per stage per call**, in order of standing savings, and
  stops at the first whose savings are not positive. The spec tries every relation.
- **Unscoped rewards for earlier frames** reach the connections to the actions that ran at those
  frames, not every connection written then.
- **The walk** fires on every update that leaves an estimate negative, so repeated negative exposures
  wire successive untried actions.
- **The subtree rule of R38** is partial: a dead neuron's own patterns are retired, which orphans their
  children onto the ledger, but a neuron that still has a parent outside the subtree is not told apart
  from one that does not.
- **The one expansion** binds an inferred call's parameters from their most frequent member, never
  from what stands at positions that have happened (D44's carry-forward).
- No backups, no persistence, no diagnostics beyond per-frame stats and table dumps.

## Two regimes

Variables are built on the residual only (D21, D33): a candidate skips every owned neighbor. An earlier
build let a new variable take neighbors out of a function's cover entry without re-measuring that entry,
which left stale entries and double-counted coverage; that regime and its `--no-claim` switch are gone.
Functions read a position a variable holds as the variable when they are collapsed (D27).

## Results

All runs on this container (4 cores), single-threaded. "claim" is the default regime, "no-claim"
`--no-claim`. Accuracy for text is next-letter accuracy of the event the machine outputs for the frame
ahead; "apex" is the cost in bits of the settled frontier per frame (D14's body, per frame); "child
voters" is the share of frames where something above the base voted.

**Text, `test.txt` (127 bytes), 5 passes, H=256, R=1**

| pass | claim: accuracy / apex / child voters | no-claim: accuracy / apex / child voters |
|---|---|---|
| 1 | 20.2% / 6.34 / 3% | 17.0% / 6.25 / 5% |
| 3 | 54.8% / 3.61 / 36% | 65.9% / 2.85 / 56% |
| 5 | 73.8% / 2.89 / 58% | 81.0% / 1.79 / 73% |

After 5 passes the stack stands 6 (claim) and 9 (no-claim) levels deep; ~8 ms per pass.

**Text, `abramov_summary.txt` (2 KB), 6 passes, H=1024, R=1**

| pass | claim: accuracy / apex / child voters | no-claim: accuracy / apex / child voters |
|---|---|---|
| 1 | 21.4% / 5.36 / 41% | 23.1% / 5.01 / 52% |
| 2 | 27.8% / 4.69 / 59% | 32.0% / 3.86 / 73% |
| 4 | 34.5% / 4.48 / 60% | 43.3% / 3.18 / 82% |
| 6 | 41.1% / 4.18 / 60% | 49.8% / 2.88 / 83% |

A bigram predictor gets about 27% here. After 6 passes: 1,598 / 2,300 neurons, 808 / 1,785 patterns,
9 / 10 levels, ~5 s / ~11 s per pass.

**Text, `abramov.txt` (32 KB), H=1024, R=1, no-claim**: first pass 33.1% (apex 3.67 bits/frame, 83% of
frames with a child voter, 3,512 neurons over 9 levels, 167 s). An earlier build of the same regime
reached 35.9% on its second pass with 13 levels; a second pass takes about 4 minutes.

**Synthetic closed loop** (`examples/world.rs`, a 7-bar cycle of moves where the right action needs
two bars of context; bigram ceiling 43%, true ceiling 71%; reward scoped to the channel and the frame;
the world runs the right action where the machine is silent): the machine takes 4 lessons in 4,000
frames and is right on 59-61% of its own calls (claim 59%, no-claim 61%), 0.15 ms per frame.

**Stocks, 5 symbols (AAPL, MSFT, NVDA, JPM, XOM), daily bars, 2,500 bars with the last 500 held out,
H=256, 7 return buckets, lessons where silent, reward = the bar's return signed by the action**

| | pass 1 train (2,000 bars) | pass 1 holdout, first sight | pass 2 holdout, second sight |
|---|---|---|---|
| claim: calls right | 50.4% | 49.9% | 53.9% |
| claim: machine vs buy&hold return | 191% vs 229% | 32% vs 60% | 74% vs 60% |
| no-claim: calls right | 50.2% | 48.8% | 51.8% |
| no-claim: machine vs buy&hold return | 139% vs 229% | 14% vs 60% | 45% vs 60% |

The machine speaks on 99.9% of symbol-bars after 10 lessons. The first-sight holdout is a coin flip; the
second pass re-runs bars the machine was already rewarded on, so its edge is memory, not
generalization. Per frame: 5-10 ms with 5 symbols; ~90 ms with 10 symbols (the earlier 10-symbol run
was at 50.4% right after 3,500 bars). Dictionary after two passes: 4,335 neurons over 19 levels
(claim), 1,791 over 9 (no-claim).

**MNIST, 14×14 binarized, 1,000 training images (100 per class) with the first 300 taught, then 500
test images, H=2048, reward +1/-1 scoped to the digit channel, two frames per image**

| | train (after the 300 lessons) | test, 500 images |
|---|---|---|
| claim | 14.6% | 17.4% |
| no-claim | 13.7% | 23.4% |

A longer run (3,000 training images, 1,000 taught, 1,000 test, earlier build) reached 20.5% on test.
Chance is 10%. The stack reaches level 4-5 on an image; 70-130 ms per image at H=2048.


## Reading the numbers

- **Text** behaves as the design intends: the frontier's cost per settled frame falls with passes,
  the stack grows a letter at a time to a dozen levels, and next-letter accuracy climbs from the
  bigram level without any supervised signal. The large file is slow because every one of its
  thousands of neurons re-covers its whole history on each call; profiling is in
  `examples/text.rs` (per-phase timings).
- **The synthetic world** confirms the forward side: the machine speaks after a handful of lessons
  and exceeds the bigram ceiling (43%) on a cycle that needs two bars of context, reaching ~60% of a
  71% maximum.
- **Stocks** on daily bars is a coin flip (49-50% of the machine's calls right), as it was for the old
  engine; being out of the market half the time, the equal-weight portfolio lags buy-and-hold over a
  bull run. Nothing here says the compression finds no structure, only that daily returns carry little.
- **MNIST** is the weak one. Under D11 the single ink neuron pools every pixel position, so the only
  carriers of digit information are shapes high enough in the stack to be digit-specific, and after a
  few thousand images the stack reaches level 4-5 (pairs, corners, short strokes). Accuracy is above
  chance (10%) but far from the old engine's position-specific readout. This is the design's bet, not an
  implementation shortcut, and it is the first thing to look at: how many levels an image needs before
  its frontier names digit-specific parts, and whether the level-above pricing lets them be bought.

## Where to look next

1. R49's match, so inferred programs are struck when they stop matching.
2. The variable position tests of D27 and classes with combinations, which the copy and equal cases
   need.
3. The two regimes above: whether the widening term should price members as it does.
4. Performance: incremental counts for the collapse (the implementation document's "pattern counts")
   instead of re-reading rows; a spatial index for neighborhoods; not scanning every neuron's history on
   a death.

# Second pass: the extended cycle test and the MNIST review

The acceptance test was the synthetic extended cycle (`apps/stocks/jobs/synthetic-extended-test.js`): the
first 12 rows of KGC repeated, binary price and volume buckets, own/out, the world running the right action
where the machine is silent, a reward scoped to the price channel and the frame. It is now
`apps/stocks/jobs/ucar-extended.js` (and `brain/ucar-core/examples/cycle.rs`, which can trace tables, bids,
mints and votes), with a test in `tests/machine.rs` that asserts more than 90% optimal decisions over the
last ten cycles.

## Bugs found and fixed

- **A neuron's activations took the level of each new bid** instead of the neuron's own level (D2), so one
  child drifted from level 1 to level 5 as its bids covered higher things, and the frame's processing order
  no longer matched its tables.
- **The add test and the margin read different counterfactuals** for a function over variables: it was
  added by its gain over what the variables saved alone in each row, and retired by a margin that charged
  what they would save alone in every row. The same pattern was added, retired and re-added every few calls.
  Both now read one formula (D30's "without f, its variables bid alone where that pays").
- **Recognition re-derived every cover from scratch**, so a candidate priced on the rows it joined (R15) lost
  them to a same-ratio rival the next call. Recognition now covers the residual of a row and keeps what the
  cover holds (§5.2), and re-centering re-measures a pattern's own entries in place (D29). The old behavior is
  `UCAR_REDERIVE=1`.
- The class collapse dropped members that happened to come after the first failing candidate in count order.

## What the cycle test showed

Measured over the last ten of 100 cycles, H=256 (`last10` in the job's output):

| configuration | optimal |
|---|---|
| full machine, R=1 | 62-65% |
| full machine, R=4 or 12 | 65-77% |
| no patterns at all, bare base neurons, R=12 | **92.5%** |
| full machine with the stack capped at one level, R=4 (probe) | 91.7% |
| functions only (no variables), stack capped, R=12 | 91.7% |
| functions only, a silent gap of 24 frames after each cycle, R=4 | **96.4%** (100 cycles); **100%** at 200 cycles |
| functions only, a gap of 12, R=1 | 90.9% (100 cycles); **100%** at 200 cycles |
| functions only, no gap, 200 cycles | 82.5% |
| functions only, gap, H=64 instead of 256 | 84% at 100 cycles, unstable after |
| full machine (variables on), same gap | 54-60% |

Two things explain the gap between the bare base and the full machine.

**1. An unbroken periodic stream is one infinite word, and the stack unrolls it.** A child sits one level
above the highest activation its bid covered (D2), and a neighbor is whatever stands uncovered within reach
at whatever level (D5). In a stream that never pauses, the top at each frame names the top of the frame
before, so the dictionary cannot close the loop: the lap of twelve patterns can never be mutually consistent
at one set of levels, so every lap a new level forms (as soon as the previous one has recurred twice, R14),
and the apex at every frame is a neuron minted within the last lap or two. It has one or two exposures, it
silences everything beneath it (D10), and when it has nothing the fall-through (R48) reaches base marginals
at odd offsets. The spec's own loop case says this: "the chain grows a level every couple of runs", and ends
only because the world "shows nothing for a while". Traced over cycles 20-30 the frontier at a fixed cycle
position changes every lap; capped at one level it settles perfectly (one function per cycle position, 51
uses each, identical rows, no churn over 25 cycles). With a gap after each cycle the word has an end, the
towers stop at the gap, and by 200 cycles every decision of the last ten cycles is optimal, at R=4 with a
gap of 24 and at R=1 with a gap of 12 (the first bar after a gap is the world's lesson, since nothing within
reach has fired). Convergence takes about 80-100 cycles at H=256 and does not hold at H=64, where the
history spans too few laps for the position-specific functions to keep their margins. This is a property of the
design on strictly periodic input, not of this implementation; real bars never repeat exactly, and text
breaks its chains at rare transitions.

**2. On a binary alphabet the variable stage is degenerate.** A parameter over two positions whose only
member ever is `vol-up` is a function in disguise: its line costs two offsets and one neuron where a
function's costs two of each, its choice costs `log2 1 = 0` per occurrence where a function's costs
`log2 |patterns|`, and agreement between two binary positions is true half the time. So the pick, which
builds variables first (D33), fills every table with eight parameters over every pair of offsets among
`−4..0`, each with a margin of hundreds of bits, overlapping on every position; a cover takes one of them
and leaves the other seven neighbors bare (a second claimant pays the bidder again, D22), and functions can
only name variables, which overlap and fail. The position-specific functions that would carry the decision
never form. With variables off they form at once. The spec's parameter is written for an alphabet where
agreement is rare and meaningful (the alternation case, 32 letters); whether its price should change for a
binary dimension, or whether a parameter should need two members, is a design question.

The text results are unchanged by the fixes (35% / 46% at four passes on the 2 KB file, spec regime /
function regime), and the synthetic 7-bar world, which is also an unbroken cycle, is at 48-59% depending on
the recognition mode.

## MNIST review

Probes on 500 training images (first 200 taught) and 300 test images at 14×14, H=1024:

| configuration | test accuracy |
|---|---|
| full machine | 18.3% |
| stack capped at one level | 17.7% |
| stack capped at two levels | 16.0% |
| functions only | 15.7% |

Neither the stack height nor the variable stage moves it, so the loss is not the tower. What votes on an
image is the uncovered frontier: pairs, corners and short strokes of ink, each a neuron that stands at every
position it occurs at (D11), so each votes a mean reward over every digit it has stood in. The whole-digit
shapes that would be specific sit at level 6-8 and rarely recur exactly, so nothing is built on them (R14).
Two frames per image, the reward scoped to the digit channel, and the teaching phase all run as the MNIST
case describes; the machine speaks on every image after the lessons; the estimates behave as R47 says (ties
during teaching go to the share, so the first answers are frequency votes; wrong answers push an estimate
negative and the walk wires the next digit at zero). Under this design a digit is recognized only once a
pattern names most of it, and that is the thing to measure next: the level at which digit-specific children
first appear and how often they recur.

## Probes left in the code

All read from the environment, all default off, none part of the design: `UCAR_MAX_LEVEL` (cap the stack),
`UCAR_NO_VARIABLES` (skip the variable stage; also the `variables: false` option), `UCAR_NO_CLAIM` (D27's
claim rule off; also `variablesClaim: false`), `UCAR_REDERIVE` (re-derive covers), `UCAR_ALL_VOTE` (covered
activations vote), `UCAR_GAP` (silent frames per cycle in the examples), and the traces `UCAR_TRACE=<id|churn>`,
`UCAR_TRACE_ROWS`, `UCAR_VOTES=<frame>`, `UCAR_MINT=<frame>`.
