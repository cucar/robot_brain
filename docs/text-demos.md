# Text & Language Demos

Applying the brain to character-sequence prediction: one channel, the letter, a byte a frame; at each frame the brain
says the byte it expects next. For the other demos see [mnist-demos.md](mnist-demos.md),
[stock-demos.md](stock-demos.md), [shape-demos.md](shape-demos.md) and [tap-demos.md](tap-demos.md).

---

## Text Sequence Learning

The brain reads a text again and again; between episodes the context is reset and what it learned is kept.

```bash
node apps/text/jobs/test.js --file abramov.txt --episodes 5
```

A file named without a folder is read from `apps/text/data`. `--iterations <n>` repeats the text within each
episode, and `--describe <letters>` prints the dictionary and the letters' neurons at the end.

**Expected output:**
```
EPISODE 1 | right 12358/33053 (37.39%) | functions 2009
EPISODE 2 | right 13336/33053 (40.35%) | functions 2692
EPISODE 3 | right 13960/33053 (42.24%) | functions 3295
EPISODE 4 | right 14733/33053 (44.57%) | functions 3805
EPISODE 5 | right 15450/33053 (46.74%) | functions 4230
right 15450/33053 | neurons 4486 alive of 24623 | patterns 4230 | heights [4478, 7, 1]
```

The text is 33,054 bytes. The share of bytes said right climbs every episode, from 37.39% to 46.74% in five, and is
still climbing at the fifth; the dictionary grows by 683, 603, 510 and 425 functions from one episode to the next. The
last line counts the living neurons by height: 4,478 at height 0, 7 at height 1 and 1 at height 2. The run takes about
4½ minutes.

---

## Short texts

The same job on a text given on the command line, one pass, with the dictionary and the neurons of the letter `h`
printed at the end:

```bash
node apps/text/jobs/test.js "the the the the the the the the the " --describe h
node apps/text/jobs/test.js theathebthecthedthefthegtheithejthek --describe h
```

| text | right | functions | neurons |
|---|---|---|---|
| `the the the the the the the the the ` | 31/35 | 4 | 260 alive of 262 |
| `theathebthecthedthefthegtheithejthek` | 20/35 | 1 | 257 alive of 258 |

Each runs in under a second.

## The alternation

The letters x and y alternating, a letter a frame, on a channel of 26 letters (the worked case of
[algorithm-xy.md](algorithm-xy.md)). Each frame prints the letter, what the brain had said for it, what stands on the
apex, and the dictionary as it changes.

```bash
node apps/text/jobs/alternation.js 33
```

**Expected output** (the end):
```
right 30/30 | neurons 27 | functions 1
alternation inference 30/30 | wrong at frames []
recognition: 1 function(s): f0 no origin [#24→#23[-1] #23→#24[-1]] cycle -> #26 | the dictionary last changed at frame 8
```

30 of 30 said right, none wrong, with one function at the end: a cycle over the two letters, the dictionary's last
change at frame 8.

The short texts and the alternation are benchmarks (see [Benchmarks](../README.md#benchmarks)).
