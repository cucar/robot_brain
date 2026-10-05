# Prices and a position

A worked case for [algorithm.md](algorithm.md): a price channel per instrument, an action that holds or trades,
and a reward that names the frames a position was held. Nothing here is normative. It follows the definitions
through what a price neuron sees, what forms over frames, and how a position comes to be taken.

---

# 1. The environment

**In.** One channel per instrument, laid out over time alone. Each declares one event dimension, `move`, whose
buckets are the bar's return quantized at the channel's resolution: a handful of buckets from a sharp fall to a
sharp rise. One frame is one bar. Channels share only time, so a neuron of one instrument sees the frontier of
another within reach in time (D5), and a pattern may name both.

**Out.** One action dimension per channel, with three base actions, `buy`, `sell` and `hold`, and `hold`
declared as the default (D34). The channel declares that its actions are not neighbors of its events (D1): the
price does not move because the machine held, so its patterns name events alone.

**Reward.** When a position closes, the environment reports its return as a reward scoped to that channel and
to the frames the position was held, `{ reward, channels: [this one], frames: the span }` (D35). The reward
dissipates linearly back over the span (R33), so the frame the position was opened in takes the least and the
frame it closed in the most.

# 2. What forms over frames

| What stands | its reach | what forms |
|---|---|---|
| a bar's move | 1 | A move sees the move before. The owner with its one neighbor is the relation (D47); recurring pairs of moves get a function and a child. |
| a pair of bars | 2 | A pair child sees the frontier two bars back: other children and the bars nothing covered. A run of three bars forms as a pair beside the bar it left out (D5). |
| a longer shape | twice what it spans (D4) | Longer shapes, and shapes across instruments: a child in one channel standing beside a child in another at offset zero. |

Variables form where the shapes vary. An offset where moves of different size have stood is a class, "a fall of
any size here" (D41); the same bucket at two offsets, a move that repeats, is a parameter (D44). Since no pattern
names an action, the dictionary describes the market and nothing the machine did.

# 3. Taking a position

Every uncovered activation of a frame, the children standing for the shape of the last bars and the bars
nothing covered, is open through its reach, and it connects to the action that ran in each later frame it is
open through (R31). The reward of a position reaches those connections over the frames it names, so a child's
estimate for `buy` is the mean return of the positions opened under it, at each distance.

**Default.** Nothing has run yet, so `hold` is output in every channel, and the frontier connects to it at a
reward of nothing.

**The walk.** Nothing explores while nothing hurts (R37). An estimate turns negative when a held position lost,
and the connection whose estimate turned negative wires the next action in the alphabet's order, `buy` after
`hold`, at a neutral estimate. It outranks the negative one and is output; the position it opens earns what it
earns, and the estimate moves.

**Selection.** Each frame, every voter on the frontier reads its connections at the offset ahead, and the
action with the largest estimate is output for each channel (R36). A child standing for a shape that has paid
under `buy` outvotes the bare bar's marginal, since its estimate rests on the one situation its pattern names
(R35). The environment executes it, reports it, and the frontier connects to what ran.

# 4. What is and is not learned

What the machine learns about prices is structure: the shapes that recur, written once each. What it learns
about trading is a mean per situation per action per distance, and nothing else: no value function, no return
and no horizon (§1.2). A shape that stops paying keeps its line as long as it describes the bars, and its
estimates say not to trade on it.
