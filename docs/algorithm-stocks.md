# Prices and a position

A worked case for [algorithm.md](algorithm.md): a price channel per instrument, an action that says whether to
own it, a world that shows the right answer wherever the machine has none, and a reward that names the frames
a position was held. Nothing here is normative. It follows the definitions through what a price neuron sees,
what forms over frames, and how a position comes to be taken, and then through one run in full: a six-bar
cycle that repeats forever.

---

# 1. The environment

**In.** One channel per instrument, laid out over time alone (D3): a bar sees the bar before it (D4), and its
row takes the bar after it a frame later (D18). Each declares one event dimension, `move`, whose buckets are
the bar's return quantized at the channel's resolution: a handful of buckets from a sharp fall to a sharp
rise, each a base neuron (D52). One frame is one bar. Channels share only time, so a neuron of one instrument
sees the apex of another at this frame and the one before (D5), and a function may name both.

There is no spatial dimension, so a function fits at one position per frame. A call is one activation of its
interneuron at the latest bar it covers, in the pose its fit read: from which node to which, and how many laps
(D37, D53). It stands for the bars it covers and inherits their relations to the outside (D10). A
bare bar costs its relation to the bar before it, one relation (D12, D13). A call costs one, one per relation
that did not come true and one per argument, and nothing else. A bar stands on the board one frame beyond the
farthest node any function reaches from its latest (D9): while the dictionary holds pairs and cycles of offset
1, that is two frames.

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
| a bar's move | 1 (D4) | A move sees the move before it in its own channel, and the moves of the other instruments at this frame and the one before (D5). Each neighbor is one relation (D34, R21). At strength 2, `{m, m'@−1}` with `m'` one bar back is a function: two nodes, one relation, origin `m` (R50). Its fit covers that one relation for the call's one, one for one, and is not taken (R24): both bars stand bare, and the pair waits in the dictionary at a worth of −1 (D57). `{m_A, m_B}` across two instruments at offset 0 is priced the same. A move after itself, `{m, m@−1}`, is a cycle of one node, and a move after another after the first, `{m, m'@−1, m after m'}`, a cycle of two (D15). |
| the first call | 1 beyond what it landed on (D4) | A call pays from three bars. A pair grows at its past end where the bar before it recurs: the node it attaches at counts it, its row holding the bar before and the bar after (D18, R50), and the flat function `{m, m'@−1, m''@−2}` covers two relations for one. A cycle pays where it reads two laps, or one lap landing on a call. The call fires once, at the latest bar, in its pose (D53), and the next bar sees it at `−1` (D5). |
| a link over a call | 1 beyond what the call stands for | At strength 2, `{m, CALL@−1}`, the call named in its pose, is a function whose fit covers the relation to the call and the call's whole cost, 2 for 1. The function that fired the call may fit at the same bar too, its node coming true on the call where the call's pose starts at that node, and cover the same (D54). Where the two measure the same the link is taken first, since its relation reads as it stands on the board (R24); where a flat function covers more for its price, it is taken instead. The link's call stands for this bar and everything the call under it stood for (D53), and expands to it all (R28). |
| a longer shape | 1 beyond what it stands for | Each further recurrence of a stretch extends the chain by one bar, the next bar's function naming the interneuron that fired a frame before, and extends the flat function by one bar at its past end, the node there counting it from its row. Where a flat function and the chain both fit, the one that covers more for its price is taken, and on a tie the link (R24). Shapes across instruments form the same way: an interneuron in one channel beside one in another at offset `0`. |

**Where the stretch stops recurring.** A bar that has not stood after the interneuron twice holds the relation
at strength 1; it is read in the index to find what already fits, and nothing is built on it (R21, R53). A
cycle or a flat function that fits at that bar and lands on the call before it covers it, 2 for 1; where none
does, the bar stands bare with its relation to the call, and the chain starts again from the next relation
that recurs. The dictionary holds one interneuron per stretch of buckets that recurred, each as long as the
recurrence, and a stretch seen at one frame and again at another is two activations of one interneuron (D11).

**A function outbid at every bar stays.** Once the move before a bar stands under a call at every bar, the bar
sees the call and not the move (D5). The function's relations are still read: the count asks for the move
or any call whose pose starts at it, and the bar saw such a call at every bar (R50). So a cycle's worth stays
positive, it fits at every bar, loses the tie to the link there, and stays in the dictionary with its
interneuron, fit and never taken, the case the evaluation document names. What is pruned is a link whose
stretch stopped recurring: nothing stands at its node any more, under a call or bare.

**After a break.** The ring's links name calls, and after a silence no call stands, so the first bars are
bare. The cycles, which never left, fit from the first bar that reads two relations, 2 for 1, and the links
fire again from the bar after, each landing on the call before it. The ring is back within a lap.

**No variable forms.** Two pairs that differ at a node, `{m, fall₁@−1}` and `{m, fall₂@−1}`, are generalized
into `{m, C@−1}`, `C` a class, priced as a replacement of the two (R52): the two cost 2 in the dictionary and
are never taken; the class 1, and never taken either, since its call would cost one and a binding for one
relation covered (D13). Nothing is saved on either, and the class does not form. A parameter tying a move to
itself would leave no fixed node, and R52 makes nothing of it. Since no function names an action, the
dictionary describes the market and nothing the machine did.

# 3. Taking a position

Every uncovered activation of a frame, the interneurons standing for the shape of the last bars and the bars
nothing covered, is open through its reach in time (D9): one frame for a bar, the length of its stretch and
one more for an interneuron. It connects to the action that ran in each later frame it is open through (R31),
at the offset its age names (D6), under its own pose (D25). Each frame's reward reaches the connections
strengthened in that frame, so an interneuron's estimate for `own` at a distance is the mean price change of
the frames owned under it in that pose at that distance.

**What learns.** An interneuron covered at age 1, by the next bar's function extending the chain, connects
once more to its coverer and to nothing else (D10); the future table refuses an interneuron as a target, so it
writes nothing. A bar covered at age 0 by the call that fires there writes nothing but that one refused
connection. So an interneuron learns the action in the occurrences where the next bar did not extend its
stretch, and a bar in the occurrences where nothing covered it. Where the interneuron at the top has no
connection for the frame ahead, what it covers speaks for it, down to the bars (R48): each says what
followed it in the frames it stood bare.

**The first lessons.** Nothing has run yet and nothing infers, so the world runs the right action in every
channel, and the apex connects to what ran with what it earned (R35). A shape that stood before a rise comes
to hold `own` at a positive estimate, one that stood before a fall `disown`, and the next time either stands on
the apex it speaks.

**The walk.** Nothing explores while nothing hurts (R37). An estimate turns negative when owning under a shape
lost, and the connection whose estimate turned negative wires the other action, `disown` after `own`, at
strength 1 and earnings 0 (R31). Its estimate is zero, so it outranks the negative one and is output; what it
earns moves the estimate.

**Selection.** Each frame, every voter on the apex proposes what its standing inferences name at the frame
ahead (R45, R46): a bar or a call. In each channel `own` and `disown` are the candidates; each voter is one
vote, split across the two by strength; a candidate's estimate is the mean of the estimates its voters named
it with, each weighted by that voter's share; and the candidate with the largest estimate is output (R47). An
interneuron standing for a shape that has paid under `own` outvotes the bare bar's marginal only by its
estimate, since that rests on the one situation its function names (R35, R47). The environment executes it,
buying, selling or doing nothing by the difference from the state it is in, reports it, and the apex connects
to what ran. Where no voter has anything for a channel, the world's lesson runs instead, and is learned the
same way.

# 4. One run: the six-bar cycle

One instrument, two buckets, `rise` and `fall`. The same six bars repeat forever: +1%, −2%, −3%, +2%, +3%,
−1%, so the moves are `rise fall fall rise rise fall`, and the right actions are `own disown disown own own
disown`. A lap is one pass over the six.

**The puzzle.** A bar sees only the bar before it, and that is not enough to know where in the lap it is: a
rise after a fall happens at bar 1 and at bar 4. But a bar does not see a raw move one bar back. It sees
whatever neuron stands there, in whatever pose, and that can carry the whole history.

**Lap 1: the first pair.** Each bar's neuron records the move before it, and takes the move after it a frame
later. `fall` at bar 2 records `rise` one back, and `fall` at bar 6 records the same, so at bar 6 the relation
fall-after-rise has strength 2 and a function starts for it (R50): nodes `rise` and `fall`, one relation,
origin the `fall`. Its fit covers that relation for the call's one, one for one, and is not taken: the pair
waits in the dictionary, and bars 5 and 6 stand bare.

**Lap 2: the three cycles, and one call for the run.** At bar 2 the pair fits again, on the `rise` at bar
1. Rise-after-fall is at strength 2, from bar 4 of lap 1 and bar 1 of this lap, shares the `rise` with the fit,
and leads back to `fall`, a node the function has, so it would close the cycle `fall` after `rise` and `rise`
after `fall` (R50, D15). The `rise` counts it: its row at bar 1 holds `fall` before it and `fall` after
it, while the `rise` at bar 4 holds `rise` after and is not counted, and the first `rise` of the run holds
nothing before, where the lap ends, which is no failure (D54): `with` 1, worth 0, and fall-after-rise worth 2.
The dictionary's worth rises by the move, from −1 for each of two pairs to 0 for the one cycle, so it joins
(D57, R54). Fit again at bar 2, it enters at the `fall` and reads back round the cycle over what the
board holds: the `rise` at 1 and the `fall` at bar 6 of lap 1, where the board ends (D9), without a failure
(D54).

| the fit at bar 2 of lap 2 | relations |
|---|---|
| A: the two relations among bars 6 to 2 | 2 |
| B: one for the call | 1 |

It is taken. Its interneuron, call it `ALT`, fires once, at the `fall` at bar 2, in the pose from `fall` round
to `fall`, one lap (D53). At bar 3 fall-after-fall is at strength 2: the count reads `fall` one back as the
move or any call whose pose starts at it, and the `fall` at bar 3 saw a bare `fall` in lap 1 and `ALT` now
(R50). That is a cycle of one node, `FF`. Its fit reads `fall` after `fall`, the node comes true on `ALT`'s
call, the fit lands on it, and the lap stops there (D54, D10): the relation and the call's cost, 2 for 1,
taken. `FF`'s call stands for bars 6 to 3, one lap of its own (D53). At bar 4 `ALT` fits again, `rise` after
`fall` landing on `FF`'s call, whose pose starts at `fall`, 2 for 1: its pose is from `rise` to `fall`, half a
lap. At bar 5 rise-after-rise is at strength 2 the same way, `RR`, and lands on `ALT`'s call, 2 for 1. At bar
6 `ALT` lands on `RR`'s call, from `fall` to `rise`, half a lap. At bar 1 of lap 3 `ALT` lands on its own call
at bar 6, and the laps add: from `rise` to `rise`, one lap; at bar 2, from `fall` to `rise`, a lap and a
half. So from bar 2 of lap 2 on, every bar's fit lands on the call before it, 2 for 1, and the apex holds one
call for the whole run, for as long as the run goes on. From lap 3 on, each bar's call is one neuron in one
pose, and no two bars share both: the six bars are six situations before any chain forms.

**Lap 3 and on: the chain.** A bar's neuron now sees the call one frame back, in its pose (D5), and a move
after a particular call in a particular pose recurs. The poses of lap 3 recur in lap 4, so at bar 3 of lap 4
`fall` after `ALT` from `fall` to `rise` is at strength 2, and `{fall, ALT@−1}`, the call named in that pose,
is a function. Its fit covers the relation to the call and the call's cost; `FF`'s fit at the same bar lands
its node on the call and covers the same (D54):

| fit at bar 3, lap 4 | coverage | price |
|---|---|---|
| `FF`, its node landing on `ALT`'s call | 2 | 1 |
| the link, `fall` after `ALT` in that pose | 2 | 1 |

They measure the same, and the link is taken first, since its relation reads as it stands on the board (R24).
The link's call fires at this bar and covers `ALT`'s call and this bar. Its interneuron stands for this bar and
everything the previous neuron stood for, so each bar adds one link: a chain. The cost per bar in a chain is
one, the standing call, whatever it stands for.

**What it gets wrong while learning.** A link exists once its pattern has been seen twice. When the chain
reaches a link no bar has learned to follow yet, the next bar's cycle lands on the call instead, 2 for 1, and
a new chain starts from there at the next lap. Chains that start at different bars are different neurons,
even when they cover the same bars. The cycles fire where the chain breaks, each in its pose, and a cycle's
connections are keyed by pose (D25), so `ALT` at bar 1 and `ALT` at bar 4, two poses, vote from two sets of
connections. What goes wrong is only what has not stood twice yet: a voter with nothing for the frame ahead
falls through to the bars under it (R48), and a bare `rise` says the average of bars 1 and 4.

**Closing the ring.** At some lap a chain runs a full lap past one of its own neurons, at bar 5 say. The bar-5
neuron is six bars back down the chain, no further than its own length, and over its length the two
expansions are the same, so the index search finds it (D36, R43): the machine fires the bar-5 neuron again
instead of allocating a new one, the new function is added to it, and the chain becomes a ring of six neurons:

| Bar | Its neuron | Says for the next bar |
|---|---|---|
| 1 | rise, after the bar-6 neuron | disown, estimate +1.9 |
| 2 | fall, after the bar-1 neuron | disown, estimate +2.9 |
| 3 | fall, after the bar-2 neuron | own, estimate +1.9 |
| 4 | rise, after the bar-3 neuron | own, estimate +2.9 |
| 5 | rise, after the bar-4 neuron | disown, estimate +0.9 |
| 6 | fall, after the bar-5 neuron | own, estimate +0.9 |

**Deciding the action.** Every bar, exactly one voter speaks: that bar's ring neuron. The move under it is
covered and silent (D10). The ring neuron has seen what came next about 135 times, always the same thing, so
its answer is right every time.

**Cleanup.** Once the ring stands, every bar sees a ring neuron one back and never a bare move. The links of
the chains that broke during learning stop recurring and are pruned (R18), and their interneurons are
decommissioned as their last activations leave the board (R38). The three cycles stay, fit at every bar
and outbid by the ring: the alternation and the two cycles of one node. After that the machine is 13 neurons:
the 4 base ones, `rise`, `fall`, `own`, `disown`, the 3 cycles, and the 6 in the ring. The dictionary is nine
functions, 10 relations, and the run costs one: the ring's call at the latest bar, and nothing outside it.

**Results.** The run of this case under the earlier rules, where a bar saw the move under a call and the pair
was credited nothing for it, was right on 92.2% of bars over 150 laps, every miss before the ring closed, for
`H` of 64, 256 and 1024. It has not been run under the rules as they stand on 2026-11-05.

# 5. What is and is not learned

What the machine learns about prices is structure: the stretches that recur, held once each. What it learns
about trading is a mean per situation per action per distance, and nothing else (§1.2). A shape that stops
paying stays in the dictionary as long as its relations recur in the bars' histories (D57, R18), and its
estimates say not to own under it.
