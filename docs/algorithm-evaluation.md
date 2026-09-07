# UCAR — the claim, the risks, and what is still open

What [algorithm.md](algorithm.md) commits to, what could go wrong with it, and what has not been decided. Each
risk states what would be done about it, so measurement has a decision attached.

---

# 1. The falsifiable claim

Nothing in the design optimizes for prediction, and nothing scores it. A pattern is charged for the neurons it
names beside the activation and never for what followed (D17); what followed it strengthens its connections,
enters no test, and is not in the file (D9). What the machine expects next is an output it hands out (§13),
and no part of the machine reads whether it came true. Prediction is therefore a pure by-product: the machine
gets better at predicting by compressing better — richer chunks are bought, patterns upstairs describe over
richer symbols, and each child's connections are the distribution of what actually followed the situation it was
bought for.

**There is no path at all by which a prediction error can change what structure exists.** If accuracy still
tracks compression, it is the by-product thesis and nothing else.

So: **prediction accuracy should track apex reduction across levels, with no part of the machine pursuing
prediction directly.** Instrument both and plot them against each other. If they move together the thesis holds
mechanically. If they do not, the coupling between compression and prediction is where to look, and it is the
assumption everything else rests on.

The standing metric is **apex units per level per frame, paired with the dictionary size that bought them**.
It should fall with exposure on recurring data: early structure is provisional, re-centering consolidates it,
and R17 takes what is left.

---

# 2. Risks

**Neither the election nor the table finds the optimum.** The election (R23) is a heuristic for the frame's
sum, and a bill takes one candidate and one retirement rather than iterating (R19). The table provably reaches
a local optimum on a fixed history (T6), and a local optimum is what it is: two patterns straddling one
cluster, each paying while the other stands, are never merged at their own level. **Diagnostic:** on a small
neuron, compare the standing file cost against an exact assignment solved offline over the same activations and
pattern count. The gap is the basin.

**The neuron never hears what it sold.** This is the design's largest deliberate omission. The election writes
the coverage set and reports nothing back (R23), so a neuron prices every pattern on what it saw, whether or
not the machine ever buys it. A pattern that describes the neuron's activations well but is consistently outbid on
the same ground — a neighbor's unit reliably takes the territory first — stays in the dictionary, keeps being
offered, and keeps paying its line in `L`. The design bets this is rare: a neuron finds itself in many
situations and wins some and loses others. **Diagnostic:** per pattern, the fraction of its bids bought over
its life. A pattern below some low fraction for a long stretch is the case the omission gets wrong. **Fallback
if it is common:** report one bit per bid, bought or not, back to the neuron on the response of its next call,
and let R17 read benefit over bought activations only. That is far cheaper than the per-neighbor
adjustment the design used to run, and it closes the case without touching what a pattern learns.

**Earlier bidders win shared past neurons.** A bid at `f` keeps a neuron at `f − 2` against a better bid at
`f + 1` (R22). The bias is on boundary neurons between chunks and it is stated as a cost. **Diagnostic:** how
often an accepted bid's tally lost exactly one slot to an earlier frame's credit, and how often that slot
would have flipped the acceptance. If the second number is not small, the alternative is re-electing the past
within the coverage set's window, which is bounded and has been rejected so far for the ripple it sends up the
stack.

**One candidate and one retirement per bill.** A neuron restructures at most one step per activation in each
direction (R19). A neuron whose situation changes wholesale — three new chunks at once — takes three of its own
activations to build them and rebuilds nothing until it fires. That is the same rhythm the machine keeps, and it
is a rate, not a cap. **Diagnostic:** the fraction of bills whose candidate paid, over a run. Near one for a
long stretch means the neuron is building as fast as it is allowed and has a backlog; near zero means the rate
is not binding.

**The offer is wider than the cover, and the election pays for it.** Every pattern that applies is sent (R18),
not only the cover, so the slot resolution sees more bids per activation than it did, and bids from one
neuron now contend with each other on the board. Cost is `O(bids · |e|)` per level per frame where bids used
to be bounded by the cover. **Diagnostic:** bids per activation against cover size, per level, and the share
of bought bids that the cover would not have offered — which is the wide offer's whole benefit, and if it is
near zero the offer can be narrowed back at no loss.

**Holding covers can hold a stale one.** R6 keeps an activation's cover unless the re-derived one is strictly
cheaper, which is what makes the bill a descent (T6). It also means an activation saved under an old table keeps
an old cover for as long as nothing beats it strictly, and two activations with one backward half can be covered
two ways. The counts a pattern re-centers on are then partly the table's past. **Diagnostic:** the share of
activations whose held cover differs from the cover R18 would derive fresh, and the cost difference. If the share
is large and the difference is zero, the hold is doing nothing but hysteresis and the tie rule could be
loosened.

**The base votes its marginal.** A base neuron on the apex expects and infers from its own connections, which is
the average over every situation it has ever fired in (D17, R35). Early in a run that is every voter there is,
so the machine's first expectations are the coarsest ones it will ever make, and its first actions are chosen
on estimates that average over situations the base neuron cannot tell apart. Coverage silences a base neuron
only once something more specific is bought over it. **Diagnostic:** the share of apex voters that are base
neurons, per frame, over exposure; it should fall as the dictionary fills. If it stays high on data that
recurs, patterns are not being bought where the base is voting, and the compression side is where to look.

**An action judged bad early is not re-tried.** A connection never leaves (R31), so the walk wires each action once
and never again (R37); an action that was unlucky on its first samples keeps that estimate and runs again only
if it becomes the least bad of a channel where everything has been tried. In a stationary world nothing is
lost. In a world where an action's worth changes, the connection cannot notice — the design's answer is that a new
child with fresh connections notices instead (R34), which holds only where structure is actually being minted
over that situation. **Diagnostic:** per channel, the share of action connections whose estimate is negative on
fewer than three exposures and which are never selected again; and, on data where an action's worth is known
to change, how many frames pass before the apex voter for that situation is a neuron minted after the change.

**An estimate moves at `1 / strength`.** With no window, a connection with a thousand exposures moves by a
thousandth of each new sample, so a base neuron's estimate on stationary-then-changed data lags for as long
as it has history. The bet is the same as above: the general case is allowed to lag because the specific
case is a younger neuron. **Diagnostic:** for a change introduced at a known frame, the estimate at the apex
against the true worth, per frame after the change, split by the age of the voting neuron. If old voters
still decide the dimension long after the change, the hierarchy is not minting where it is needed, and the
compression side is where to look.

**Several children per activation may not halve the level above.** A neuron used to promote at most one child,
which is what made T11's halving argument work — and the halving is what D14's doubling reach is derived from,
what T12's depth bound rests on, and what keeps `|O|` constant across levels. A cover of `m` patterns promotes
up to `m` children at one coordinate (D5, R18), and the wide offer lets patterns outside the cover be bought
too, so a level can be *wider* than the one below it wherever activations decompose into several chunks. Nothing
caps `m`; what bounds it is that each extra child is another line and another set of charges, and the
election stops buying the moment one does not pay. **Diagnostic:** bought bids per activation and units per
level per frame, against the halving T11 assumes. If levels stop thinning, D14's reach schedule is calibrated
against an invariant that no longer holds, and the reach has to be derived from measured spacing instead.

**Far placement is lossy, and the loss compounds with height.** An offset is kept to one significant binary
digit (D15), so a neighbor 11 frames back is written as 8 and expanded to 8 (R27). Each level down adds its own
group's slack, so a high unit places the base symbols of its farthest neighbors only to within the sum of the
groups along the path — its forward half can say what comes next but not, above a few levels, exactly when.
The file says so (§2); what it does not say is how deep the stack stays useful before its far placements are
too coarse for the prediction (§13) or a motor program (R30) to act on, which is a cap on useful depth that
nothing declares. **Diagnostic:** per level, the placement error of expanded base symbols against where they
fired, split by offset group; and the level above which the forward half's base placements no longer win a
dimension. If that level is low, the coarse groups are earning their reach on the backward side and losing it
on the forward, and a finer offset alphabet above some level is the fallback.

**Siblings agree until one is bought alone.** Two children bought at one coordinate see the same forward
neighbors, so their connections agree until the first activation where one is bought and the other is not (D21). Two
patterns always bought together never diverge at their own level; the level above is expected to merge them
(§4 of the remarks). **Diagnostic:** for pairs of children of one neuron, the overlap of the frames they were
bought in against the overlap of their connections. Pairs high on both for a long stretch are the merge the level
above has not made.

**A newly minted child expects little for a while.** Its connections start forming the frame after its mint
(R13) and then only in the frames it is bought in (D17), so until it has been bought a few times it expects
from one or two exposures and infers little but the default. The parent it was minted from is covered whenever
the child is bought, so nothing speaks for that situation with much history until the child has some.
**Diagnostic:** frames from a child's mint to the first time its expectation wins a dimension, against how
often it is bought.

**Nothing retires a pattern for a useless future.** A pattern is priced on what it names that did not fire
beside it, and never on what its child was followed by (D17, R17). So a child whose connections are worthless keeps
its line as long as its parent's pattern covers, and the machine's expectation from it is noise. Nothing
clears a bad connection: the connections keep every symbol that ever followed, and the claim is that the vote
dilutes what it cannot clear — a neuron whose connections spread over many symbols splits its one unit thinly
across them, so it decides
a dimension only where no sharper voter contends — and that a pattern whose *backward* half is good is worth
its line regardless. A diluted voter still places something in every dimension it has ever seen, so it never
falls silent. **Diagnostic:** expected symbols per apex unit that did not arrive, per level, and the share of
dimensions decided by a voter whose winning symbol held under a tenth of its unit. If either does not fall
with exposure, dilution is not enough and the consumer of the output is getting noise the machine could have
withheld.

**The readout is unvalidated, and the position now lives outside the symbol.** Compressing harder can produce a
worse classifier, because a readout may be living on exactly the position-and-class-specific duplicates that
compression deletes — and D8 deletes them by construction rather than incidentally. The information is not
lost: it moved into the activation coordinates the body states. But a readout reading bare symbol identity
sees a translation-invariant bag and loses every bit of *where*, so it has to consume `(symbol, coordinate)`
pairs. A regression here will look like the compression was wrong when it was the decode. The readout gate in
[algorithm-implementation.md](algorithm-implementation.md) is the check.

**History size and reach sensitivity.** Every decision is exact with respect to the last `H` activations and blind
beyond them. `H` too small and patterns form on coincidences and the stack deepens faster than the evidence
warrants; too large and a neuron follows a moving situation slowly and keeps more structure than earns its
keep. Reach too small and no chunk spans what recurs; too large and every neighborhood is mostly noise at
build time. Measure both early and jointly — they interact through `|e|`, not through R4, whose denominator is
the same at every offset (R11). **Diagnostic:** how often the outermost offset is named against offset 0, swept
over `H`. If the outer reaches stay empty at every `H`, the reach is bigger than the data supports and
evidence is not what is limiting it. Sweep depth against `H` in the same runs — T13 makes the two move
together, and conflating them is easy.

**Boundary flicker on stationary input.** R4 holds at equality, which stops a slot flip-flopping when the same
population is re-read, but the ring is a FIFO and its population moves at every bill. A slot whose count sits
at the boundary follows the activations entering and leaving, naming it raises the pattern's price wherever it
is absent, that can drop the pattern out of a cover, and the smaller population re-decides every other slot.
The claim is that on stationary input this is flicker around a fixed point, confined to boundary slots, with an
amplitude that does not grow with run length — a claim about noise, which nothing in the rules proves. **This is
the standing test.** Per pattern per bill: slot flips against the slot's distance from the boundary,
`2 · count − n − 1`, and cover changes per bill for the cascade. Expected: flips concentrated within a step or
two of the boundary, at a rate that settles once the ring is full and does not drift. A flip rate that rises with
run length, or flips far from the boundary, is the churn engine and a bug. Early tests are also decided by very
little evidence, so read the same numbers over the first thousand frames and again in steady state.

**One-shot builds.** A seed present in a handful of activations gives a population of that handful, and the
collapse over a small population names most of what it holds. Re-centering largely defuses it — the pattern is
pulled toward whatever recurs, or starves. **Fallback if it still churns:** require the seed's population to
span at least two activations before a candidate is priced. Exact, and costs one recurrence of latency.

**Shared patterns fit every position worse than tuned ones would.** D8 pools activations from everywhere into one
pattern, so a pattern describes statistics that genuinely differ by position and fits each of them worse. That
is a real cost and it is paid in charges, which is the body half of `L` — the dictionary half falls in
exchange, and R12 is what weighs the two. The design commits to the trade being worth it and offers no way to
buy back position-specificity except declaring a coarse position as a *neuron* dimension. **Diagnostic:**
charges per activation against dictionary size, before and after, on the same data.

**The cover pass is a greedy set cover, not a nearest-neighbor lookup.** One scan of the table per round, and
a round is one pattern taken (R18). Cost is `O(|cover| · |table| · |O⁻|)`, and `|cover|` is exactly the
quantity the multi-child risk above says is unbounded. **Diagnostic:** cover-pass scans per activation against
cover size, per level.

**Routing cost at the base.** `|O|` is held constant across levels by construction (D14), but its value is set
by the reach and the base density, and the cover pass prices every pattern against it every frame.
**Diagnostic:** scan volume per level, against `|O|`.

**An early partition can freeze.** A pattern never acquires a neighbor another pattern of the same cover already
holds — the slot's population excludes those activations entirely (R4) — so patterns grow into the residual and
never into each other. That is what stops two patterns converging, and T6 rests on it, and it also means a bad
early split of one chunk across two patterns is not repaired by re-centering. It can only be repaired by one
of them retiring and the other growing into what it left, one per bill. **Diagnostic:** how often a retirement
is followed within `H` activations by a surviving pattern growing into the vacated neighbors. If it is rare, the
partition is sticky and R14 is carrying more of the load than intended.

**Election slack, bounded but unmeasured.** R23 is ratio-greedy weighted set cover, so its slack against the
best cover buildable from the same bids is bounded by `H(n)` and no better
([algorithm-remarks.md](algorithm-remarks.md) §17). The bound is worst-case and
says nothing about the slack on real frames, and since apex-units-per-frame is the headline metric, slack and
real structure are conflated in it. **Diagnostic:** solve one small window exactly (ILP) and compare, which
locates the realized slack inside the `H(n)` ceiling.

**The composition gap.** Both scopes price in one currency against one `L`, and the neuron's prices and the
machine's are meant to differ (D12). What remains is that candidates are *generated* locally: a demand no
neuron proposes is a symbol the election never gets to consider, and no neuron proposes one whose value lies in
what it would let a *different* neuron stop paying for. Distinct from election slack, which measures the
election against a perfect election over the same bids; this measures propose-then-elect against optimizing
dictionary and frames together. **Diagnostic:** over a short run on one small level, compare the file this
design writes against the file a joint optimization over the same activations produces. That gap decides whether
contraction should stay purely a buyer or start supplying candidates back into the tables it covered — the
constituents of one chunk each build their own near-duplicate of it today, which is where a constructive
variant would pay first.

---

# 3. Open questions

**Neighborhood space at higher levels.** Above level 0 the neighbors are patterns, and the per-dimension alphabet
grows as patterns are created, so the space expands with the structure. What no longer expands is `|O|`:
adjacency is a reach at every level (D4), and D14 sets that reach precisely to hold the expected neighbor count
fixed as the level thins. So the open measurement is whether the invariant holds in practice, since it rests
on T11's halving being close to what contraction actually achieves. **Diagnostic:** neighbors per activation, per
level, against the constant the invariant predicts.

**Parallelism.** The per-neuron passes are independent across neurons and could run at once. Re-centering makes
them slightly less independent, and D8 makes the neuron population smaller and each neuron busier — every
position sharing a type folds into one table, so the parallelism available shifts from across-neuron toward
across-activation, and re-centering becomes the contended point. The election is not sequential: R23 is two
decisions and a settling, each over every slot or every bid at once, with nothing revisited. The only ordering
constraint is that a level's bills and offers must all be in before its election runs, and both fall inside
one frame (T14). With nothing reported back, no bill waits on an election. On much larger inputs than MNIST all
of this needs revisiting.

**Asymmetric reach, and isotropic growth.** Backward and forward reach both emerge from the vote, bounded by
the same reach. Whether one reach is right — "how much do I need to recognize myself" and "how far do I need to
expect" are different questions — is unresolved, and it matters more now that the connections are the whole of
the machine's expectation. One reach is the committed choice; separate reaches are the fallback if diagnostics
show neighborhoods consistently reaching the bound in one direction only. **The same doubt applies across
activation dimensions**: D14 grows every one of them by the same factor, which holds only if contraction thins
them equally, and a channel whose patterns chunk harder in time than in space would want otherwise.
**Diagnostic:** mean spacing per activation dimension, per level, against the isotropic prediction.

**Whether a coarse voter should count as one.** The vote at the base gives every apex activation one unit per
dimension whatever level it stands at (§13, R36). A level-4 unit that expanded to forty base symbols and a base neuron
that expects one symbol are then equal voters in the dimension they share, and the level-4 unit's expectation about
that dimension rests on a situation the base neuron's does not. Level was taken out of the vote because ranking by
level let compression override reward; whether some other reading of the voter — the strength behind its connection,
the exposures its estimate rests on — should weight it is not decided. **Diagnostic:** per dimension, how often the
winner was placed by a base voter over a pattern voter that disagreed, and which was right.

**The cross-neuron seam.** R4's abstention teaches a pattern not to name what another pattern of the *same
cover* holds. Across neurons nothing teaches it: a neighbor some other neuron's unit reliably covers is still
present, still wins its majority, and stays in the neighborhood, paying its slot in the line. This is the same
omission as the neuron never hearing what it sold, seen from the neighborhood's side rather than the bid's,
and it is the same bet. The one-bit fallback above would not close it; closing it takes the per-neighbor report
the design removed. **Diagnostic:** per pattern, the share of its named neighbors that its bought bids were
never credited for, over its life.

**R33's shaping is ahead of the implementation; the cycle and the arithmetic are not.** The spec says a reward
carries an optional channel set and an optional frame span, dissipates linearly over that span, and enters the
estimate of the action connection at the age each distance names. The current brain runs the two-frame cycle
as R29 says — the action inferred at one frame fires at the next with its reward attached — and folds rewards
into the estimate exactly as R31 says, one exposure, one share, weighted `1 / strength`. What it does not do is
shape them: it attaches the frame's reward whole to the action that ran in that frame, on every open age, which
is R33's span of one, and a minted pattern is pre-wired with `rewards[age − distance]` on a second path.
Neither scopes and neither ramps. **Diagnostic:** with a span-of-one reward, whether the estimate of an action
that ran in an earlier frame moves at all — under R33 it must not.

**The forward side of the code is the design, with four deliberate differences.** A connection is a lifetime
total on the neuron, strengthened on observation, never weakened, never collapsed; the estimate is the exact running
mean; the walk wires the next untried action on a negative mean; covered ages are silenced; and the vote at the base
normalizes each voter to one unit per dimension and distance, events winning by share and actions by the
share-weighted mean estimate, with no level anywhere in it. All of that is D20, R31, R36 and R37 as written. What
differs: the code wires the declared default at birth at strength 1, where R35 lets it run and be learned; action
neurons vote for actions in the code, where R35 lets only events choose; the code wires every level's connections to
the base set instead of the level below its own (D17), and so needs no expansion (R27, R30); and it shapes no reward
(above).
[algorithm-implementation.md](algorithm-implementation.md) lists the changes.
