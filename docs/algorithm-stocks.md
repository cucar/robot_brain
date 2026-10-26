# Prices and a position

A worked case for [algorithm.md](algorithm.md): a price channel per instrument, an action that says whether to
own it, a world that shows the right answer wherever the machine has none, and a reward that names the frames
a position was held. Nothing here is normative. It follows the definitions through what a price neuron sees,
what forms over frames, and how a position comes to be taken.

---

# 1. The environment

**In.** One channel per instrument, laid out over time alone (D3), with radius `R = 1` (R4): a bar sees the
bar before it. Each declares one event dimension, `move`, whose buckets are the bar's return quantized at the
channel's resolution: a handful of buckets from a sharp fall to a sharp rise, each a base neuron (D52). One
frame is one bar. Channels share only time, so a neuron of one instrument sees the apex of another within
reach in time (D5), and a function may name both.

There is no spatial dimension, so a bare record is two numbers, the neuron and the frame (D12, D13). A call's
record is two numbers, plus one per relation that did not come true and one per value.

**Out.** One action dimension per channel, with two base actions, `own` and `disown`: the state the machine
wants to be in for that instrument. The environment holds the state it is in, compares the two, and buys, sells
or does nothing; the machine never names a trade.

**Policy.** The machine declares its action policy (D51) as the least bad: an instrument is owned or disowned
every day, and where the machine is silent the world runs its own action. It declares its neighborhood policy
(D50) with every entry that has an action off in the past table, since the price does not move because the
machine owns, and in the future table only event to action and interneuron to action on. Everything below
follows from that and nothing is declared for it: no action is anyone's neighbor, so no function names one, so
no action is ever covered and every action on the apex is base; a connection reaches base actions only, so an
inference expands to nothing and contradicts nothing; actions write no connections and never vote; and every
exposure of one action after one situation pools on one connection, which is the connection a reward scoped to
that channel pays (R33).

**Teaching.** Where the machine says nothing for an instrument, the world runs the right action for that frame
itself, `own` where the bar rose and `disown` where it fell, and rewards it (R35). That runs all the time,
beside the trading: a situation is taught until the machine speaks in it, and from then on what the machine
says is what runs.

**Reward.** Every frame, for every instrument, the environment reports the change in price since the last
frame, signed by the action that stood over it, positive for `own` where the price rose and for `disown` where
it fell, as a reward scoped to that channel and to that one frame, `{ reward, channels: [this one], frames: 1 }`
(D35). It arrives with the next frame's prices and with the report of the action that ran, so the connection to
the action is strengthened and takes the reward in one write (R29, R31).

# 2. What forms over frames

| What stands | its reach | what forms |
|---|---|---|
| a bar's move | 1 (D4) | A move sees the move before it in its own channel, and the moves of the other instruments at this frame and the one before (D5). Each neighbor is one relation (D34, R21). Held in two rows, `{m, m'@−1}` is a function: A is two bare records, 4; B the call's record, 2; its entry one relation and two neurons, 3; its margin at two rows 1 (D30). `{m_A, m_B@0}` across two instruments is priced the same. Where the rows of the two intersect, grouping merges them into `{m, m'@−1, m_B@0}`, 6 for 2 (R50). |
| an interneuron over a pair | 1 beyond the farthest bar it stands for (D4): 2 | The next bar sees the interneuron at `−1` as itself, and the bar under it at `−1` (D5, D47). `{m, PAIR@−1}`, held in two rows, lands on everything the pair's call covers and takes it back (R23): A is the bar's record and the call's record, 4; B is 2. Its interneuron stands for three bars and expands to them (R28). |
| a longer shape | 1 beyond what it stands for | Each further recurrence of the stretch extends the chain by one bar, the next bar's function naming the interneuron that fired a frame before. An interneuron also sees the bar before its stretch, and a function from its history may name it. Shapes across instruments form the same way: an interneuron in one channel beside one in another at offset `0`. |

**Where the stretch stops recurring.** A bar that has not stood after the interneuron twice holds the relation
in one row; it is read in the index to find what already fits, and nothing is built on it (R21, R53). A pair
`{m, m'@−1}` with its own interneuron lands on one of what the chain's call covers, and is credited nothing
for it (R23), so it measures 2 for 2 and is not taken. The bar stands bare, and the chain starts again from the
next pair that recurs. So the dictionary holds one interneuron per stretch of buckets that recurred, each as
long as the recurrence, and a stretch seen at one frame and again at another is two activations of one
interneuron (D11).

**No variable forms.** Two pairs that differ at a node, `{m, fall₁@−1}` and `{m, fall₂@−1}`, are generalized
into `{m, C@−1}`, `C` a class, priced as a replacement of the two (R52): the two cost 6 in the dictionary and 2
per occurrence; the class 5 and 3 per occurrence, the member that stood written as a value (D13). Each specific
function has at least two occurrences, so the class is never cheaper and does not join. A move that repeats,
`{m, m@−1}`, is a pair whose two nodes name one neuron. A parameter tying the two would leave no fixed node in
the latest frame, and R52 makes nothing of it. Since no function names an action, the dictionary describes the
market and nothing the machine did.

# 3. Taking a position

Every uncovered activation of a frame, the interneurons standing for the shape of the last bars and the bars
nothing covered, is open through its reach in time (D9): one frame for a bar, the length of its stretch for an
interneuron. It connects to the action that ran in each later frame it is open through (R31), at the offset
its age rounds to (D6). Each frame's reward reaches the connections strengthened in that frame, so an
interneuron's estimate for `own` at a distance is the mean price change of the frames owned under it at that
distance.

**What learns.** An interneuron covered at age 1, by the next bar's function extending the chain, connects
once more to its coverer and to nothing else (D10); the future table refuses an interneuron as a target, so it
writes nothing. A bar covered at age 0 writes nothing but that one refused connection. So an interneuron learns
the action in the occurrences where the next bar did not extend its stretch, and a bar in the occurrences where
nothing covered it. Where the interneuron at the top has no connection for the frame ahead, what it covers
speaks in its place, down to the bars (R48): each says what followed it in the frames it stood bare.

**The first lessons.** Nothing has run yet and nothing infers, so the world runs the right action in every
channel, and the apex connects to what ran with what it earned (R35). A shape that stood before a rise
comes to hold `own` at a positive estimate, one that stood before a fall `disown`, and the next time either
stands on the apex it speaks.

**The walk.** Nothing explores while nothing hurts (R37). An estimate turns negative when owning under a shape
lost, and the connection whose estimate turned negative wires the other action, `disown` after `own`, at
strength 1 and earnings 0 (R31). Its estimate is zero, so it outranks the negative one and is output; what it
earns moves the estimate.

**Selection.** Each frame, every voter on the apex proposes what its standing inferences place at the frame
ahead (R45, R46). In each channel `own` and `disown` are the candidates; each voter is one vote, split across the
two by strength; a candidate's estimate is the mean of the estimates its voters placed it with, each weighted by
that voter's share; and the candidate with the largest estimate is output (R47). An interneuron standing for a
shape that has paid under `own` outvotes the bare bar's marginal only by its estimate, since that rests on the
one situation its function names (R35, R47). The environment executes it, buying, selling or doing nothing by
the difference from the state it is in, reports it, and the apex connects to what ran. Where no voter has
anything for a channel, the world's lesson runs instead, and is learned the same way.

# 4. What is and is not learned

What the machine learns about prices is structure: the stretches that recur, held once each. What it learns
about trading is a mean per situation per action per distance, and nothing else (§1.2). A shape that stops
paying stays in the dictionary as long as its margin is positive over the bars it describes (D30, R18), and its
estimates say not to own under it.
