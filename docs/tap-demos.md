# Tap Demos

Tap while go, then done (the worked case of [algorithm-loop.md](algorithm-loop.md)): one channel over time alone, an
event dimension `cue` (go, stop) and an action dimension `act` (tap, done). For the other demos see
[stock-demos.md](stock-demos.md), [text-demos.md](text-demos.md), [shape-demos.md](shape-demos.md) and
[mnist-demos.md](mnist-demos.md).

---

## Tap while go, then done

A stretch shows go in frames 0, 1 and 2, stop in frame 3, and no cue after.

- **Teaching:** the world runs tap itself in frames 1, 2 and 3 and done in frame 4, each rewarded. Then it shows
  nothing and runs nothing for 8 frames, longer than any reach. Six stretches.
- **Test:** three stretches of 3, 5 and 2 go frames. The world shows the cues, runs nothing itself, and executes what
  the brain outputs. A tap in the frame after a go and done in the frame after a stop are rewarded 1; any other
  action −1.

The neighborhood policy is the stocks one: no action is anyone's neighbor, and only events and interneurons connect,
to actions alone. Nothing runs where every candidate's estimate is negative.

```bash
node apps/tap/jobs/tap.js 6 8
```

The arguments are the teaching stretches (6) and the silent frames after each stretch (8). Each frame prints the cue,
what ran, what stands on the apex (a guessed activation marked `?`), and the brain's output; then the test table, the
dictionary and every living neuron with its connections.

**Expected output** (the end of the test table):
```
right 37/37
longest reach in time seen: 4 (silence 8)
guessed activations that stood: []
dictionary: 3 functions
```

| | |
|---|---|
| right in the test | 37/37 |
| functions at the end | 3 |
| run time | under 1 s |

This is the `tap` benchmark; its full output is saved in [benchmarks/output/tap.out](benchmarks/output/tap.out)
(see [Benchmarks](../README.md#benchmarks)).
