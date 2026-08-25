# UCAR — the claim, the risks, and what is still open

What [algorithm.md](algorithm.md) commits to, what could go wrong with it, and what has not been decided. Each
risk states what would be done about it, so measurement has a decision attached.

---

# 1. The falsifiable claim

Nothing in the design optimizes for prediction. The one test prices file length; prediction error is priced
only because, under a predictively coded history, it *is* file length. The machine gets better at predicting by
compressing better — richer chunks propagate, and entries upstairs describe over richer symbols.

So: **prediction accuracy should track apex reduction across levels, with no part of the machine pursuing
prediction directly.** Instrument both and plot them against each other. If they move together the thesis holds
mechanically. If they do not, the coupling between compression and prediction is where to look, and it is the
assumption everything else rests on.

The standing metric is **apex units per level per frame, paired with the dictionary size that bought them**.
It should fall with exposure on recurring data: early structure is provisional, re-centering consolidates it,
and the pruning pass retires what is left.

---

# 2. Risks

**Neither the election nor the table finds the optimum.** Greedy (R28) is an approximation to R22, and a bill
takes one improvement step rather than iterating (R19). Both are bounded, and neither is exact.
**Diagnostic:** the election's gap is already covered by the ILP comparison below.

**The table is never settled.** A bill absorbs its evidence, makes at most one structural decision and
re-centers once, so a bin can sit with a server that is no longer its closest entry until something reads its
distances. Lloyd converges to a local optimum and this design does not even run Lloyd to convergence —
deliberately, since every bill moves the counts that a converged table would have been converged against. Add
and delete are what move the table between basins, and the add is triggered only by a negative contribution
(R13), so a neuron whose situations are all explained sits in whatever basin it landed in. **Diagnostic:** on
a small neuron, compare the standing service cost against an exact assignment solved offline over the same
bins and entry count, and separately against the same
table iterated to a fixed point. The first gap is the basin, the second is what not iterating costs.

**Neighborhood growth over long spans.** `|e|` can grow as an entry accumulates stable slots across `2·reach_t + 1`
frames, and cost grows with it, so an entry can price itself out by learning too much. **Fallback:** cap `|e|`,
dropping lowest-vote neighbors first. The vote already ranks them.

**The readout is unvalidated, and the position now lives outside the symbol.** Compressing harder can produce a
worse classifier, because a readout may be living on exactly the position-and-class-specific duplicates that
compression deletes — and D8 deletes them by construction rather than incidentally. The information is not
lost: it moved into the activation coordinates the history states. But a readout reading bare symbol identity
sees a translation-invariant bag and loses every bit of *where*, so it has to consume `(symbol, coordinate)`
pairs. A regression here will look like the compression was wrong when it was the decode. The readout gate in
[algorithm-implementation.md](algorithm-implementation.md) is the check.

**History size and radius sensitivity.** Every decision is exact with respect to the last `H` observations and
blind beyond them. `H` too small and entries form on coincidences and the stack deepens faster than the
evidence warrants; too large and a neuron follows a moving situation slowly and keeps more structure than earns
its keep. Radius too small and no chunk spans what recurs; too large and every neighborhood is mostly noise at
mint time. Measure both early and jointly — they interact through `|e|`, not through R4, whose denominator is
the same at every offset (R11). **Diagnostic:** how often the outermost offset is named against offset 0, swept
over `H`. If the outer reaches stay empty at every `H`, the radius is bigger than the data supports and
evidence is not what is limiting it. Sweep depth against `H` in the same runs — T16 makes the two move
together, and conflating them is easy.

**Cold-start churn.** Early tests are decided by very little evidence. Re-centering is the main defense, but
measure churn over the first thousand frames and again in steady state.

**One-shot mints.** A single neighborhood far enough from a settled entry can out-bid the opening cost by
itself. Re-centering largely defuses it — the neighborhood is pulled toward whatever recurs, or the entry
starves. **Fallback if it still churns:** require the win set to span at least two distinct bins. Exact, and
costs one recurrence of latency.

**A use the election took can settle negative, by construction.** The election accepts on the backward half —
all a bid has seen (R21) — and the price lands on the whole span (D13), so the forward misses arrive after the
acceptance and nothing retracts it (R27). The claim is that this state is transient or terminal, never an
equilibrium: re-centering votes out neighbors wrong more often than right, the add test splits off
distinguishable demand, and an entry negative in sum is retired (remarks, on D13). **Diagnostic:** the
fraction of observations settling negative, per neuron, over exposure. It should fall to the minority tail of entries
positive in sum; an entry persistently negative in sum that survives the pruning pass is a bug in the pass,
not a basin.

**The adjustment is a snapshot of one board, reused against many tables.** It is frozen at its frame (R2) while
everything priced against it keeps moving: neighborhoods re-center, entries are minted and retired, and the
covering unit that made a neighbor uncredited may itself be long gone. The record still says that neighbor was
taken. So an entry can be under-credited for territory that has since been released, until the observations
carrying the stale adjustment are evicted. **Diagnostic:** how often a covered neighbor's coverer is retired
while the adjustment is still held, and what the margins would have been without those adjustments.

**Adjustment cost at high fan-out.** Every activation reads the board once at its bill, sized by its
neighborhood, and a neuron brings every one of a frame's activations to a single bill. The per-read size is
held down by D15's invariant, which fixes the expected neighbor count at every level; what grew is the number of
reads per neuron per frame. Each is a lookup into a structure the machine already maintains, and the bill scans
the routing table once for all of them — so it should disappear into the pass it sits in. **Diagnostic:** read
volume against routing cost, per level, and against activations per bill.

**What subsumption does to a neuron's own structure.** A neuron reliably covered by a neighbor's unit prices
every entry at near zero, so its table is pruned to the normal within `H` of its own firings and it stops
bidding — which is correct while the coverage holds and leaves it with nothing when the coverage lapses. It
then rebuilds from its own history, which is intact (R29), but it is silent for `H` firings first.
**Diagnostic:** how many of its own firings a neuron whose coverer is retired takes to bid again.

**Shared entries fit every position worse than tuned ones would.** D8 pools observations from everywhere into
one entry, so an entry describes statistics that genuinely differ by position and fits each of them worse. That
is a real cost and it is paid in corrections, which is the history half of `L` — the dictionary half falls in
exchange, and R12 is what weighs the two. The design commits to the trade being worth it and offers no way to
buy back position-specificity except declaring a coarse position as a *neuron* dimension. **Diagnostic:**
corrections per activation against dictionary size, before and after, on the same data.

**Routing cost at the base.** `|O|` is held constant across levels by construction (D15), but its value is set
by the radii and the base density, and routing prices every entry against it every frame. A radius large enough to be
useful at depth is inherited by no other level — each has its own — but the base still pays for whatever was
declared. **Diagnostic:** routing scan volume per level, against `|O|`, against the radii.

**A level's bare majority takes a slot from a level that knew better.** Within a level the collapse settles a
contest on claims alone and is revisable until written, so nothing there rests on an estimate. Across levels
the cascade gives the higher level first refusal, and promotion is a finding about *description*: nothing
establishes that the better describer of a region is the better predictor of one slot inside it. Two level-3
claims agreeing take the slot even where the level-0 neuron owning that dimension would have been right. The
cascade only steps down when a level cannot decide, never when it decides badly. **Diagnostic:** count
cross-level contests and how often the holder was right against how often the level below it was.

**Election slack, and now with no bound at all.** R28 is a heuristic for R22, and unlike the greedy it replaced
it carries no approximation guarantee — two fixed passes over a static per-slot rule have no `ln n` backstop.
Since apex-units-per-frame is the headline metric, slack and real structure are conflated, and the slack is now
unmeasured rather than merely unmeasured-but-bounded. **Diagnostic:** solve one small window exactly (ILP) and
compare — and compare the per-slot rule against a greedy pass over the same bids, which is the cheaper of the
two comparisons and isolates what the change cost.

**The composition gap.** Narrowed, not closed. Both scopes now price in one currency against one `L`, and the
adjustment removes the double-counting that made them disagree. What remains is that candidates are *generated*
locally: a demand no neuron proposes is a symbol the election never gets to consider, and no neuron proposes
one whose value lies in what it would let a *different* neuron stop paying for. Distinct from election slack,
which measures the election against a perfect election over the same bids; this measures propose-then-elect
against optimizing dictionary and frames together. **Diagnostic:** over a short run on one small level, compare
the file this design writes against the file a joint optimization over the same observations produces. That gap
decides whether contraction should stay purely inhibitory or start supplying candidates back into the routing
tables it covered — the constituents of one chunk each mint their own near-duplicate of it today, which is
where a constructive variant would pay first.

---

# 3. Open questions

**Neighborhood space at higher levels.** Above level 0 the neighbors are patterns, and the per-dimension alphabet
grows as patterns are created, so the space expands with the structure even though D5 holds each
`(dimension, position)` to a single state. What no longer expands is `|O|`: adjacency is a radius at every level
(D4), and D15 sets that radius precisely to hold the expected neighbor count fixed as the level thins. So the open
measurement is narrower than it was — not whether `|O|` explodes, but whether the invariant holds in practice,
since it rests on T9's halving being close to what contraction actually achieves. **Diagnostic:** neighbors per
observation, per level, against the constant the invariant predicts.

**Parallelism.** The per-neuron passes are independent across neurons and could run at once. Re-centering makes
them slightly less independent, and D8 makes the neuron population smaller and each neuron busier — every
position sharing a type now folds into one routing table, so the parallelism available shifts from
across-neuron toward across-activation, and re-centering becomes the contended point. The election is not
sequential: R28 is two decisions and a settling, each over every slot or every bid at once, with nothing
revisited. The adjustment
adds one ordering constraint and no message traffic: a level's bets must all be in before its election runs, and
that election must have run before any of its bills read the board. Both fall inside one frame (T14) and neither
crosses a level, but on much larger inputs than MNIST all three judgments need revisiting.

**Asymmetric reach, and isotropic growth.** Backward and forward reach both emerge from the vote, bounded by
the same radius. Whether one radius is right — "how much do I need to recognize myself" and "how far can I
reliably predict" are different questions — is unresolved. One radius is the committed choice; separate radii
are the fallback if diagnostics show neighborhoods consistently reaching the bound in one direction only. **The
same doubt applies across activation dimensions**: D15 grows every one of them at `2^(1/dim)`, which holds only
if contraction thins them equally, and a channel whose patterns chunk harder in time than in space would want
otherwise. **Diagnostic:** mean spacing per activation dimension, per level, against the isotropic prediction.

**The pattern does not learn what the price learns.** A neighbor some other unit reliably covers earns an entry
nothing, and the contributions say so (D13) — but the collapse votes on presence, so the neighbor stays in the
neighborhood and the line keeps paying for it, and the entry can only shed it by dying whole. The tallies to
change this already exist: a credited-presence vote is `presence − covered`, per slot, off D22. What it would
cost is the seam — letting the candidate's collapse read election outcomes lets the board shape a *claim* (R29
guards evidence, and a claim is not evidence, so it may be legal), and it changes what T1's center means.
Whether structure should learn creditedness, or only prices should, is open.
