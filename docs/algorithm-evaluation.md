# UCAR — the claim, the risks, and what is still open

What [algorithm.md](algorithm.md) commits to, what could go wrong with it, and what has not been decided. Each
risk states what would be done about it, so measurement has a decision attached.

---

# 1. The falsifiable claim

Nothing in the design optimizes for prediction, and nothing scores it. A pattern is charged for the neurons it
names beside the firing and never for what followed (D17); its forward half is measured by the collapse,
enters no test, and is not in the file (D9). What the machine expects next is an output it hands out (§13),
and no part of the machine reads whether it came true. Prediction is therefore a pure by-product: the machine
gets better at predicting by compressing better — richer chunks are bought, patterns upstairs describe over
richer symbols, and each one's forward half is the consensus of what actually followed the situation it names.

**There is no path at all by which a prediction error can change what structure exists.** If accuracy still
tracks compression, it is the by-product thesis and nothing else.

So: **prediction accuracy should track apex reduction across levels, with no part of the machine pursuing
prediction directly.** Instrument both and plot them against each other. If they move together the thesis holds
mechanically. If they do not, the coupling between compression and prediction is where to look, and it is the
assumption everything else rests on.

The standing metric is **apex units per level per frame, paired with the dictionary size that bought them**.
It should fall with exposure on recurring data: early structure is provisional, re-centering consolidates it,
and the retire pass takes what is left.

---

# 2. Risks

**Neither the election nor the table finds the optimum.** The election (R23) is a heuristic for the frame's
sum, and a bill takes one candidate and one retirement rather than iterating (R19). The table provably reaches
a local optimum on a fixed history (T6), and a local optimum is what it is: two patterns straddling one
cluster, each paying while the other stands, are never merged at their own level. **Diagnostic:** on a small
neuron, compare the standing file cost against an exact assignment solved offline over the same firings and
pattern count. The gap is the basin.

**The neuron never hears what it sold.** This is the design's largest deliberate omission. The election writes
the coverage set and reports nothing back (R23), so a neuron prices every pattern on what it saw, whether or
not the machine ever buys it. A pattern that describes the neuron's firings well but is consistently outbid on
the same ground — a neighbor's unit reliably takes the territory first — stays in the dictionary, keeps being
offered, and keeps paying its line in `L`. The design bets this is rare: a neuron finds itself in many
situations and wins some and loses others. **Diagnostic:** per pattern, the fraction of its bids bought over
its life. A pattern below some low fraction for a long stretch is the case the omission gets wrong. **Fallback
if it is common:** report one bit per bid, bought or not, back to the neuron on the response of its next call,
and let the retire test read benefit over bought firings only. That is far cheaper than the per-neighbor
adjustment the design used to run, and it closes the case without touching what a pattern learns.

**Earlier bidders win shared past neurons.** A bid at `f` keeps a neuron at `f − 2` against a better bid at
`f + 1` (R22). The bias is on boundary neurons between chunks and it is stated as a cost. **Diagnostic:** how
often an accepted bid's tally lost exactly one slot to an earlier frame's credit, and how often that slot
would have flipped the acceptance. If the second number is not small, the alternative is re-electing the past
within the coverage set's window, which is bounded and has been rejected so far for the ripple it sends up the
stack.

**One candidate and one retirement per bill.** A neuron restructures at most one step per firing in each
direction (R19). A neuron whose situation changes wholesale — three new chunks at once — takes three of its own
firings to build them and rebuilds nothing until it fires. That is the same rhythm the machine keeps, and it
is a rate, not a cap. **Diagnostic:** the fraction of bills whose candidate paid, over a run. Near one for a
long stretch means the neuron is building as fast as it is allowed and has a backlog; near zero means the rate
is not binding.

**The offer is wider than the cover, and the election pays for it.** Every pattern that applies is sent (R18),
not only the cover, so the slot resolution sees more bids per activation than it did, and bids from one
neuron now contend with each other on the board. Cost is `O(bids · |e|)` per level per frame where bids used
to be bounded by the cover. **Diagnostic:** bids per activation against cover size, per level, and the share
of bought bids that the cover would not have offered — which is the wide offer's whole benefit, and if it is
near zero the offer can be narrowed back at no loss.

**Holding covers can hold a stale one.** R6 keeps a firing's cover unless the re-derived one is strictly
cheaper, which is what makes the bill a descent (T6). It also means a firing folded under an old table keeps
an old cover for as long as nothing beats it strictly, and two firings with one backward half can be covered
two ways. The counts a pattern re-centers on are then partly the table's past. **Diagnostic:** the share of
firings whose held cover differs from the cover R18 would derive fresh, and the cost difference. If the share
is large and the difference is zero, the hold is doing nothing but hysteresis and the tie rule could be
loosened.

**Exploration waits for the first bought pattern.** A voter is an apex activation reading the pattern it was
promoted from, and a base neuron on the apex has none (§10.3), so an action slot nothing reaches runs the
declared default (R35). Before any pattern is bought, every slot is that slot, and the machine runs the default
only. The walk (R37) begins on the first pattern that is bought and lands on the apex, and until then no reward
teaches anything. On event-dense input that is a few frames; on sparse input it could be long. **Diagnostic:**
the frame of the first non-default action, against the frame of the first bought pattern. **Fallback:** give
base neurons a forward record of their own, marginal over every situation, and let them vote where nothing
above them does — which is the earlier design's neuron-level connection, and the reason it was dropped is
that it votes the general case into every specific one (R36).

**Several children per firing may not halve the level above.** A neuron used to promote at most one child,
which is what made T11's halving argument work — and the halving is what D14's doubling reach is derived from,
what T12's depth bound rests on, and what keeps `|O|` constant across levels. A cover of `m` patterns promotes
up to `m` children at one coordinate (D5, R18), and the wide offer lets patterns outside the cover be bought
too, so a level can be *wider* than the one below it wherever firings decompose into several chunks. Nothing
caps `m`; what bounds it is that each extra child is another line and another set of charges, and the
election stops buying the moment one does not pay. **Diagnostic:** bought bids per activation and units per
level per frame, against the halving T11 assumes. If levels stop thinning, D14's reach schedule is calibrated
against an invariant that no longer holds, and the reach has to be derived from measured spacing instead.

**Siblings share a future.** Every pattern covering a firing sums the same forward half (D21), so children
bought together carry one expectation between them and diverge only as their coverage diverges. Two patterns
always bought together never diverge at their own level; the level above is expected to merge them (§4 of the
remarks). **Diagnostic:** for pairs of patterns of one neuron, the overlap of the firings they cover against
the overlap of their forward halves. Pairs high on both for a long stretch are the merge the level above has
not made.

**Nothing retires a pattern for a useless future.** A pattern is priced on what it names that did not fire
beside it, and never on what it named ahead (D17, R17). So a pattern whose forward half is worthless keeps its
line as long as its backward half covers, and the machine's expectation from it is noise. The claim is that
the collapse handles this without a test — a forward event slot with no majority drops out, so a useless
expectation becomes silence rather than a wrong symbol — and that a pattern whose *backward* half is good is
worth its line regardless. **Diagnostic:** expected symbols per apex unit that did not arrive, per level. If
that share does not fall with exposure, the collapse is not clearing bad slots and the consumer of the output
is getting noise the machine could have withheld.

**The readout is unvalidated, and the position now lives outside the symbol.** Compressing harder can produce a
worse classifier, because a readout may be living on exactly the position-and-class-specific duplicates that
compression deletes — and D8 deletes them by construction rather than incidentally. The information is not
lost: it moved into the activation coordinates the history states. But a readout reading bare symbol identity
sees a translation-invariant bag and loses every bit of *where*, so it has to consume `(symbol, coordinate)`
pairs. A regression here will look like the compression was wrong when it was the decode. The readout gate in
[algorithm-implementation.md](algorithm-implementation.md) is the check.

**History size and reach sensitivity.** Every decision is exact with respect to the last `H` firings and blind
beyond them. `H` too small and patterns form on coincidences and the stack deepens faster than the evidence
warrants; too large and a neuron follows a moving situation slowly and keeps more structure than earns its
keep. Reach too small and no chunk spans what recurs; too large and every neighborhood is mostly noise at
build time. Measure both early and jointly — they interact through `|e|`, not through R4, whose denominator is
the same at every offset (R11). **Diagnostic:** how often the outermost offset is named against offset 0, swept
over `H`. If the outer reaches stay empty at every `H`, the reach is bigger than the data supports and
evidence is not what is limiting it. Sweep depth against `H` in the same runs — T13 makes the two move
together, and conflating them is easy.

**Cold-start churn.** Early tests are decided by very little evidence. Re-centering is the main defense, but
measure churn over the first thousand frames and again in steady state.

**One-shot builds.** A seed present in a handful of firings gives a population of that handful, and the
collapse over a small population names most of what it holds. Re-centering largely defuses it — the pattern is
pulled toward whatever recurs, or starves. **Fallback if it still churns:** require the seed's population to
span at least two firings before a candidate is priced. Exact, and costs one recurrence of latency.

**Shared patterns fit every position worse than tuned ones would.** D8 pools firings from everywhere into one
pattern, so a pattern describes statistics that genuinely differ by position and fits each of them worse. That
is a real cost and it is paid in charges, which is the history half of `L` — the dictionary half falls in
exchange, and R12 is what weighs the two. The design commits to the trade being worth it and offers no way to
buy back position-specificity except declaring a coarse position as a *neuron* dimension. **Diagnostic:**
charges per activation against dictionary size, before and after, on the same data.

**The cover pass is a greedy set cover, not a nearest-neighbor lookup.** One scan of the table per round, and
a round is one pattern taken (R18). Cost is `O(|cover| · |table| · |O⁻|)`, and `|cover|` is exactly the
quantity the multi-child risk above says is unbounded. **Diagnostic:** cover-pass scans per firing against
cover size, per level.

**Routing cost at the base.** `|O|` is held constant across levels by construction (D14), but its value is set
by the reach and the base density, and the cover pass prices every pattern against it every frame.
**Diagnostic:** scan volume per level, against `|O|`.

**An early partition can freeze.** A pattern never acquires a neighbor another pattern of the same cover already
holds — the slot's population excludes those firings entirely (R4) — so patterns grow into the residual and
never into each other. That is what stops two patterns converging, and T6 rests on it, and it also means a bad
early split of one chunk across two patterns is not repaired by re-centering. It can only be repaired by one
of them retiring and the other growing into what it left, one per bill. **Diagnostic:** how often a retirement
is followed within `H` firings by a surviving pattern growing into the vacated neighbors. If it is rare, the
partition is sticky and the build is carrying more of the load than intended.

**Election slack, with no bound at all.** R23 is a heuristic for the objective, and unlike the greedy it
replaced it carries no approximation guarantee — two fixed passes over a static per-slot rule have no `ln n`
backstop. Since apex-units-per-frame is the headline metric, slack and real structure are conflated.
**Diagnostic:** solve one small window exactly (ILP) and compare — and compare the per-slot rule against a
greedy pass over the same bids, which is the cheaper of the two comparisons and isolates what the change cost.

**The composition gap.** Both scopes price in one currency against one `L`, and the neuron's prices and the
machine's are meant to differ (D12). What remains is that candidates are *generated* locally: a demand no
neuron proposes is a symbol the election never gets to consider, and no neuron proposes one whose value lies in
what it would let a *different* neuron stop paying for. Distinct from election slack, which measures the
election against a perfect election over the same bids; this measures propose-then-elect against optimizing
dictionary and frames together. **Diagnostic:** over a short run on one small level, compare the file this
design writes against the file a joint optimization over the same firings produces. That gap decides whether
contraction should stay purely a buyer or start supplying candidates back into the tables it covered — the
constituents of one chunk each build their own near-duplicate of it today, which is where a constructive
variant would pay first.

---

# 3. Open questions

**Neighborhood space at higher levels.** Above level 0 the neighbors are patterns, and the per-dimension alphabet
grows as patterns are created, so the space expands with the structure. What no longer expands is `|O|`:
adjacency is a reach at every level (D4), and D14 sets that reach precisely to hold the expected neighbor count
fixed as the level thins. So the open measurement is whether the invariant holds in practice, since it rests
on T11's halving being close to what contraction actually achieves. **Diagnostic:** neighbors per firing, per
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
expect" are different questions — is unresolved, and it matters more now that the forward half is the whole of
the machine's expectation. One reach is the committed choice; separate reaches are the fallback if diagnostics
show neighborhoods consistently reaching the bound in one direction only. **The same doubt applies across
activation dimensions**: D14 grows every one of them by the same factor, which holds only if contraction thins
them equally, and a channel whose patterns chunk harder in time than in space would want otherwise.
**Diagnostic:** mean spacing per activation dimension, per level, against the isotropic prediction.

**Whether the output needs an owner per slot.** The prediction is the union over the apex of what each unit
expects (R27). Two units expecting different symbols at one base slot both go out. That is more information
than a vote, and it is the consumer's problem to reduce. Whether any consumer the machine is built for — a
readout, a downstream machine, a scorer — actually wants the set or wants one symbol is not decided, and if the
latter the cascade the assertion used to run is the obvious reduction to hand it. **Diagnostic:** how often
the union at a base slot holds more than one symbol, per level of the units that placed them.

**The cross-neuron seam.** R4's abstention teaches a pattern not to name what another pattern of the *same
cover* holds. Across neurons nothing teaches it: a neighbor some other neuron's unit reliably covers is still
present, still wins its majority, and stays in the neighborhood, paying its slot in the line. This is the same
omission as the neuron never hearing what it sold, seen from the neighborhood's side rather than the bid's,
and it is the same bet. The one-bit fallback above would not close it; closing it takes the per-neighbor report
the design removed. **Diagnostic:** per pattern, the share of its named neighbors that its bought bids were
never credited for, over its life.

**R33 is ahead of the implementation.** The spec says a reward carries an optional channel set and an optional
frame span, dissipates linearly over that span, and lands beside the action neighbor in every open activation's
firing. The current brain does none of that shape: it holds connections apart from forward tallies, hands every
open age the frame's reward whole and unscoped on one path and attributes `rewards[age − distance]` on another,
and neither ramps. **Diagnostic:** with a span-of-one reward, whether a slot at offset `d > 1` moves at all —
under R33 it must not.

**The forward record has not been implemented as one object.** The spec folds connections into the forward
half (D20, R31); the brain still keeps two. Until they are one, every place the spec says "action slot" the
code has a connection that may or may not agree with the tally beside it. **Diagnostic:** for a pattern and an
action, the count on its forward tally against the strength on its connection, which under D20 are one number.
