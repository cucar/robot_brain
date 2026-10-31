# Prices and a position

A worked case for [algorithm.md](algorithm.md): a price channel per instrument, an action that says whether to
own it, a world that shows the right answer wherever the machine has none, and a reward that names the frames
a position was held. Nothing here is normative. It follows the definitions through what a price neuron sees,
what forms over frames, and how a position comes to be taken, and then through one run in full: a six-bar
cycle that repeats forever.

---

# 1. The environment

**In.** One channel per instrument, laid out over time alone (D3): a bar sees the bar before it (D4). Each
declares one event dimension, `move`, whose buckets are the bar's return quantized at the channel's
resolution: a handful of buckets from a sharp fall to a sharp rise, each a base neuron (D52). One frame is one
bar. Channels share only time, so a neuron of one instrument sees the apex of another at this frame and the one
before (D5), and a function may name both.

There is no spatial dimension, so a function fits at one place per frame, and its origin is its node at the
current bar: a call fires at the latest bar it covers and covers back from there (D15, D37). A bare bar counts
its name, 1, and the relation to the bar before it, 1 (D12, D13). A call counts its name, 1, plus one per
relation that did not come true and one per value.

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
| a bar's move | 1 (D4) | A move sees the move before it in its own channel, and the moves of the other instruments at this frame and the one before (D5). Each neighbor is one relation (D34, R21). At strength 2, `{m', m}` with `m'` one bar back is a function: two nodes, one offset, origin `m` (R50). Its fit covers the relation and both names, 3, for a call of 1 (D22), so it is accepted and gets an interneuron, which fires at `m`'s bar (R16, D53). `{m_A, m_B}` across two instruments at offset 0 is priced the same. |
| an interneuron over a pair | 1 beyond the farthest bar it stands for (D4): 2 | The next bar sees the interneuron at `−1` and nothing under it (D5, D10). At strength 2, `{PAIR, m}` is a function whose fit covers the relation, the bar's name and the call's count, 3 for 1. The plain pair fits at the same bar too: its first node comes true on the call whose origin it is, lands on it and covers the same count, 3 for 1 (D54). The two measure the same, and the link is taken first, since its relation reads as it stands on the board (R24); the pair then lands on nothing uncovered. The link's interneuron stands for three bars and expands to them (R28). |
| a longer shape | 1 beyond what it stands for | Each further recurrence of the stretch extends the chain by one bar, the next bar's function naming the interneuron that fired a frame before. An interneuron also sees the bar before its stretch, and a function from its history may name it. Shapes across instruments form the same way: an interneuron in one channel beside one in another at offset `0`. |

**Where the stretch stops recurring.** A bar that has not stood after the interneuron twice holds the relation
at strength 1; it is read in the index to find what already fits, and nothing is built on it (R21, R53). The
pair function for that bar's move is placed, its first node landing on the chain's call as the origin's neuron
(D54), covers 3 for 1, and is accepted, so the bar stands in a pair and the chain starts again from there. The
dictionary holds one interneuron per stretch of buckets that recurred, each as long as the recurrence, and a
stretch seen at one frame and again at another is two activations of one interneuron (D11).

**What stops being seen is pruned.** A pair that a link outbids at every bar is still placed there and
declined, and nothing records the verdict (D57). What prunes it is the bar's own history: once the move before
it stands under a call at every bar, the bar sees the call and not the move (D5), the pair's relation is no
longer observed, its `with` slides out of the history, its worth falls below zero, the relation leaves, the
function goes, and its interneuron is decommissioned once the last window over it closes (R18, R38). The same
happens to the links of a chain that stopped recurring. A pair outbid at some bars and bare at others keeps
recurring and stays.

**No variable forms.** Two pairs that differ at a node, `{fall₁, m}` and `{fall₂, m}`, are generalized into
`{C, m}`, `C` a class, priced as a replacement of the two (R52): the two cost 6 in the dictionary and 1 per
occurrence; the class 5 and 2 per occurrence, the member that stood counted as a value (D13). Each specific
function has at least two occurrences, so the class is never cheaper and does not join. A move that repeats,
`{m, m}`, is a pair whose two nodes name one neuron. A parameter tying the two would leave no fixed node, and
R52 makes nothing of it. Since no function names an action, the dictionary describes the market and nothing the
machine did.

# 3. Taking a position

Every uncovered activation of a frame, the interneurons standing for the shape of the last bars and the bars
nothing covered, is open through its reach in time (D9): one frame for a bar, the length of its stretch for an
interneuron. It connects to the action that ran in each later frame it is open through (R31), at the offset
its age names (D6). Each frame's reward reaches the connections strengthened in that frame, so an interneuron's
estimate for `own` at a distance is the mean price change of the frames owned under it at that distance.

**What learns.** An interneuron covered at age 1, by the next bar's function extending the chain, connects
once more to its coverer and to nothing else (D10); the future table refuses an interneuron as a target, so it
writes nothing. A bar covered at age 0, as the origin of the call that fires there, writes nothing but that
one refused connection. So an interneuron learns
the action in the occurrences where the next bar did not extend its stretch, and a bar in the occurrences where
nothing covered it. Where the interneuron at the top has no connection for the frame ahead, what it covers
speaks in its place, down to the bars (R48): each says what followed it in the frames it stood bare.

**The first lessons.** Nothing has run yet and nothing infers, so the world runs the right action in every
channel, and the apex connects to what ran with what it earned (R35). A shape that stood before a rise comes
to hold `own` at a positive estimate, one that stood before a fall `disown`, and the next time either stands on
the apex it speaks.

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

# 4. One run: the six-bar cycle

One instrument, two buckets, `rise` and `fall`. The same six bars repeat forever: +1%, −2%, −3%, +2%, +3%,
−1%, so the moves are `rise fall fall rise rise fall`, and the right actions are `own disown disown own own
disown`. A lap is one pass over the six.

**The puzzle.** A bar sees only the bar before it, and that is not enough to know where in the lap it is: a
rise after a fall happens at bar 1 and at bar 4. But a bar does not see a raw move one bar back. It sees
whatever neuron stands there, and that neuron can carry the whole history.

**Lap 1: the first pair.** Each bar's neuron records the move before it. `fall` at bar 2 records `rise` one
back, and `fall` at bar 6 records the same, so at bar 6 the relation rise-then-fall has strength 2 and a
function starts for it (R50): nodes `rise` and `fall`, one offset, origin the `fall`. It is placed at bar 6,
covers the relation and both names, 3, for a call of 1, is accepted, and gets an interneuron, which stands for
bars 5 and 6 and fires at bar 6 (D53). The `fall` at bar 6 is covered at age 0.

**Lap 2: every pair.** The other three pairs reach strength 2 in turn, fall-then-rise at bar 1, fall-then-fall
at bar 3, rise-then-rise at bar 5, and each is accepted the same way. A pair whose first node stands as the
origin of the previous pair's call lands on that call and covers it, 3 for 1 (D54). By the end of the lap every
bar is the origin of one pair call, and each pair's interneuron stands on the apex where the next bar can see
it.

**Lap 3 and on: the chain.** A bar's neuron now sees the call one frame back, and nothing under it (D5).
"My move, after the neuron from one bar back" reaches strength 2 at lap 3. Its fit covers the relation, the
bar's name and the call's count; the plain pair's fit at the same bar lands its first node on the call through
its origin and covers the same (D54):

| fit at a bar, lap 3 | coverage | price |
|---|---|---|
| the pair, its first node landing on the call as its origin | 3 | 1 |
| the link, this move after the previous call | 3 | 1 |

They measure the same, and the link is taken first, since its relation reads as it stands on the board (R24);
the pair then lands on nothing uncovered and is declined. The link's call fires at this bar and covers the
previous call and this bar. Its interneuron stands for this bar and
everything the previous neuron stood for, so each bar adds one link: a chain. The count per bar in a chain is
one number, the link's call.

**Why it makes mistakes while learning.** A link exists once its pattern has been seen twice. When the chain
reaches a link no bar has learned to follow yet, the next bar stands in a plain pair and a new chain starts from
there. Chains that start at different bars are different neurons, even when they cover the same bars. Where a
chain breaks, the lone bar votes its average over every situation it has stood in, and the abandoned chain end
votes what its few sightings taught it. Those are the wrong answers while the chain is being built.

**Closing the ring.** At some lap a chain runs a full lap past one of its own neurons, at bar 5 say. The bar-5 neuron
is six bars back down the chain, and over those six bars the two expansions are the same, so the index search
finds it (D36, R43): the machine fires the bar-5 neuron again instead of allocating a new one, the new function
is added to it, and the chain becomes a ring of six neurons:

| Bar | Its neuron | Says for the next bar |
|---|---|---|
| 1 | rise, after the bar-6 neuron | disown, estimate +1.9 |
| 2 | fall, after the bar-1 neuron | disown, estimate +2.9 |
| 3 | fall, after the bar-2 neuron | own, estimate +1.9 |
| 4 | rise, after the bar-3 neuron | own, estimate +2.9 |
| 5 | rise, after the bar-4 neuron | disown, estimate +0.9 |
| 6 | fall, after the bar-5 neuron | own, estimate +0.9 |

**Deciding the action.** Every bar, exactly one neuron votes: that bar's ring neuron. The move under it is
covered and silent (D10). The ring neuron has seen what came next about 135 times, always the same thing, so
its answer is right every time.

**Cleanup.** Once the ring stands, every bar sees a ring neuron one back and never a bare move, so the four
pairs' relations stop being observed, their `with` slides out of the bars' histories, and they are pruned
(R18); the links of the chains that broke during learning go the same way. Their interneurons are
decommissioned as the last windows over them close (R38). After that the machine is 10 neurons: the 4 base
ones, `rise`, `fall`, `own`, `disown`, and the 6 in the ring. The dictionary is six functions of two nodes and
one offset, 18 numbers, and each bar costs 1.

**Results.** The run of this case under the earlier rules, where a bar saw the move under a call and the pair
was credited nothing for it, was right on 92.2% of bars over 150 laps, every miss before the ring closed, for
`H` of 64, 256 and 1024. It has not been run under the rules as they stand on 2026-10-31.

# 5. What is and is not learned

What the machine learns about prices is structure: the stretches that recur, held once each. What it learns
about trading is a mean per situation per action per distance, and nothing else (§1.2). A shape that stops
paying stays in the dictionary as long as its relations recur in the bars' histories (D57, R18), and its
estimates say not to own under it.
