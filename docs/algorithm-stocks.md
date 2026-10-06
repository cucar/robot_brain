# Prices and a position

A worked case for [algorithm.md](algorithm.md): a price channel per instrument, an action that says whether to
own it, a world that shows the right answer wherever the machine has none, and a reward that names the frames
a position was held. Nothing here is normative. It follows the definitions through what a price neuron sees,
what forms over frames, and how a position comes to be taken.

---

# 1. The environment

**In.** One channel per instrument, laid out over time alone, with radius `R = 1` (D1): a bar sees the bar
before it. Each declares one event dimension, `move`, whose
buckets are the bar's return quantized at the channel's resolution: a handful of buckets from a sharp fall to a
sharp rise. One frame is one bar. Channels share only time, so a neuron of one instrument sees the frontier of
another within reach in time (D5), and a pattern may name both.

**Out.** One action dimension per channel, with two base actions, `own` and `disown`: the state the machine
wants to be in for that instrument. The environment holds the state it is in, compares the two, and buys, sells
or does nothing; the machine never names a trade. The channel declares that its actions are not neighbors of
its events (D1): the price does not move because the machine owns, so its patterns name events alone.

**Teaching.** Where the machine says nothing for an instrument, the world runs the right action for that frame
itself, `own` where the bar rose and `disown` where it fell, and rewards it (§3.5). That runs all the time,
beside the trading: a situation is taught until the machine speaks in it, and from then on what the machine
says is what runs.

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
estimate for `own` is the mean return of the frames owned under it, at each distance.

**The first lessons.** Nothing has run yet and nothing infers, so the world runs the right action in every
channel, and the frontier connects to what ran with what it earned (R35). A shape that stood before a rise
comes to hold `own` at a positive estimate, one that stood before a fall `disown`, and the next time either
stands on the apex it speaks.

**The walk.** Nothing explores while nothing hurts (R37). An estimate turns negative when owning under a shape
lost, and the connection whose estimate turned negative wires the other action, `disown` after `own`, at a
neutral estimate. It outranks the negative one and is output; what it earns moves the estimate.

**Selection.** Each frame, every voter on the frontier reads its connections at the offset ahead, and the
action with the largest estimate is output for each channel (R36). A child standing for a shape that has paid
under `own` outvotes the bare bar's marginal, since its estimate rests on the one situation its pattern names
(R35). The environment executes it, buying, selling or doing nothing by the difference from the state it is in,
reports it, and the frontier connects to what ran. Where no voter has anything for a channel, the world's
lesson runs instead, and is learned the same way.

# 4. What is and is not learned

What the machine learns about prices is structure: the shapes that recur, written once each. What it learns
about trading is a mean per situation per action per distance, and nothing else: no value function, no return
and no horizon (§1.2). A shape that stops paying keeps its line as long as it describes the bars, and its
estimates say not to own under it.
