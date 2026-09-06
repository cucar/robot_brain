# UCAR — theorems and commentary

Everything here elaborates [algorithm.md](algorithm.md) and none of it is normative. **T** is a theorem: a
claim that follows from the definitions and rules, stated with its argument. Everything else is commentary —
why the design is shaped the way it is, what the alternative would have cost, and worked examples. Items are
keyed by the D, R or section they belong to, in the order the specification introduces them.

---

# 1. The machine

**On D2 — why a child sits at offset zero.** A neighborhood is written relative to the parent's activation, so the
child sits at offset zero by construction. A centroid over what the child covers would leave the level above
differencing coordinates from origins its neighbors were never measured against.

**On D4 — a neighborhood is a box.** Adjacency is a conjunction, so something far away in one dimension is not
a neighbor however close it is in another, and no dimension can rescue what another has ruled out.

**On D4 — receptive fields grow with depth two ways at once.** Through composition, because a neighbor is itself
a chunk. And through the reach, because each level's units are sparser than the one below and have to reach
further to find each other at all. Neither is declared per level: the first is what a pattern *is*, and D14
derives the second from the first.

**On D4 — why adjacency is not declared.** A conjunction over shared activation dimensions already says
everything a visibility declaration would, and it says it without a table to maintain: what a channel is laid
out over settles which channels it can be adjacent to.

**On D5 — many activations per coordinate.** The base bound is the input's: one symbol per channel at each point
of its layout. Above the base there is no bound at all, because a neuron covers its activation with a set of
patterns and each of them may promote a child (R18). What replaced the bound is not a weaker version of it but
a different kind of rule — the only exclusivity left is credit (D7), and credit is about paying, not naming.

**On D5 — the two halves are simultaneous, not sequential.** An action is not a reply appended to a frame.
Reading it as one puts the action a frame later than it is and breaks every offset measured across the two
sides.

**On D6 — ages in a single neuron.** A neuron active at frames 10 and 12 is at ages 0 and 2 in frame 10 + 2,
then 1 and 3 in the next, and so on.

**On D6 — an activation is an event, not a state.** It decides everything it will ever decide in the frame it
fires, on a backward half that is already whole (D17). What it does afterwards is connect and speak. Connecting is
transcription rather than judgement — the arriving neighbors and rewards strengthen the neuron's connections,
where the *next* expansion reads them. Speaking is reading the neuron's own connections, and it changes
nothing the neuron holds.

**On D6 — why the window is exactly `reach_t`.** The last forward neighbor fires at offset `reach_t`, and an
action's reward arrives in the frame the action runs in (R29), so nothing can land after that frame and there
is no frame to hold open for. An earlier draft paid the reward one frame after the action and had to keep every
activation open one frame longer for that single arrival; the two-frame cycle removes the frame and the reason.

**On D7 — three consequences of having no rest value.**

- **Blank regions cost nothing** — no activation, no history, no dictionary, no compute.
- **A frame is a set** — variable-size, containing only what happened.
- **Absence discriminates in one direction only** — a pattern naming a filled disk is measurably wrong on a
  hollow circle, because the interior neurons it names did not fire and each is charged (D16). The reverse is
  not an error: a circle-pattern on a filled disk covers what it names and leaves the interior in the residual,
  where it costs the same lines it always would. **What separates them is coverage, not penalty** — the
  disk-pattern covers more of the disk, so it wins the disk, and the circle-pattern is not charged for losing
  it.

**On D7 — where sparsity comes from.** Whether a 0 pixel is a black event or nothing-to-see is the encoder's
choice, made where the input is produced. A dimension where something always happens is simply always active;
nothing depends on sparsity, it only profits from it.

**On D7 — two units naming one neuron is not a defect.** The decoder expanding both gets the neuron twice and
the neuron fires once; a set does not care how many times it is told a member. What would be a defect is
paying two lines where one would do, and that is what exclusive credit prevents: a bid is worth only what
would otherwise stand as its own line (R23), so a unit whose only contribution is a neuron another unit already
delivers cannot clear its price. Naming is free, paying is exclusive.

**On D8 — what position-specificity would buy is fit, and that is what it costs.** One pattern serving every
position describes statistics that differ by position, so it fits each worse than a position-tuned pattern
would, and the neighbors it names wrongly are literally the charges in the file (D16). The dictionary half of
`L` falls and the history half rises. **R12 adjudicates exactly that trade**, pattern by pattern, and no price
anywhere changes for it to do so: an activation costs 1 before and after, because the alphabet loses precisely
the factor the coordinate gains.

# 2. The objective

**On D9 — why the file holds nothing about the future.** The forward half of a pattern used to be a claim the
file scored: a unit asserted what would follow, a wrong assertion was a correction, and the corrections were a
term of `L`. Three things were wrong with it. Nothing priced the corrections — no test read them and nothing
was ever retired for predicting badly — so the term sat in the objective and decided nothing. The bid had to
carry a half it could not be scored on. And a slot two units both predicted needed an owner, which needed a
resolution, which needed a second exclusivity beside coverage. Removing the claim removes all three. What is
left is a dictionary coder over the backward window, and what the machine expects next is an output it hands
to whoever is reading, scored there or not at all.

**On D9 — the file is not a log of how the machine got there.** A pattern that goes is not a line the file must
keep alive: the run is simply re-encoded without that symbol, and the frames it used to cover are expressed by
whatever the dictionary now offers, with charges where that is worse. An unbounded file costs nothing because
nothing was ever going to write it.

**On D11 — where compression is actually paid out.** A unit firing at `h` names `[h − reach_t, h]`, so writing
it discharges up to `reach_t + 1` frames of the run at once. That is why a wider reach is worth paying a wider
line for, and it is why the residual is free and a child is not.

**On D12 — why `covers` counts what turned up rather than what the pattern names.** Take a neuron `x` whose
pattern names `{a, b, c}` backward, in a frame where `a` and `b` fired, `c` did not, and an unnamed `m` did.

```
without the unit   state x, a, b, m                                            4 symbols
with the unit      1 for the unit, which expands to x, a, b, c
                   charges: turn off c;  m stands as its own line              3 symbols
                                                              true saving      1
```

`|O| − d = 3 − 2 = 1`, which is the saving. Now let only `a` fire, so `O = {a}` and `d = 2`: the file states
`x, a` for 2 symbols without the unit and pays `1 + 2` with it, a saving of `−1`, and `|O| − d = 1 − 2` gives
exactly that. **Counting what the pattern names would give `|e| − d = 3 − 2 = +1` and report a saving where the
file got longer** — `b` and `c` would be credited as delivered *and* charged as absent, netting nothing, so a
name that never fires would be free to hold. The two error types are the ones being told apart: naming wrongly
costs a symbol and delivers nothing, while a neighbor left unnamed costs its own line where stating it flat
cost a line, so it is free either way.

> **T1 — A fixed-length code is not a real file length, and no test can tell.** D10 prices a symbol at one
> however often it is used. That is not what a decoder pays: naming a symbol out of a dictionary that grows
> without bound (D3) costs about `log |alphabet|` bits, and that rises over the run. Write the true length as
> `c · L`, with `c` the bits one symbol name currently costs.
>
> **Every test reads the sign of a difference, never a length.** `margin = benefit − cost` (R12), and both
> terms are counted in the same symbols, so the true margin is `c · margin`. `c > 0`, so the sign is the same
> one. The dictionary term and the history term scale together because both are counts of the same symbols —
> if they did not, the constant would not cancel and this would fail.
>
> **So counting symbols and counting bits answer every question the machine asks the same way.** That is what
> buys the integer arithmetic: no estimator, no smoothing, no boundary correction anywhere that decides what
> structure exists. It is a property of a *fixed-length* code, and re-pricing the file by how often each symbol
> occurs gives it up — the constant stops cancelling and probabilities do set costs, which is why
> [forgetting.md](forgetting.md) is a document of its own.

**On D12 — the two sums are what the two mechanisms work against, and that is the whole division of labor.**
The election works on the history half over a given dictionary — it is priced in exactly that sum, for the
frames it can see, though it does not minimize it (§11.3). The one test decides the dictionary half, pattern
by pattern (R12). Neither can do the other's job: the election cannot create or destroy a symbol, and a neuron
cannot see what its symbol saved.

**On D12 — two readers, two numbers, and why that is economics rather than an inconsistency.** The neuron is
deciding what to hold and what to offer, over every situation it has been in; the machine is deciding what to
buy, for the one window in front of it. The neuron's number says whether a pattern pays over its own history;
the machine's says whether a bid pays on a board where earlier frames' credit stands and other neurons' bids
contend. A pattern the neuron holds because it pays in most of its activations can lose at the election in this
one, and a pattern the neuron's own cover passed over can be the machine's best purchase, because the machine's
residual is not the neuron's (R18, R22). An earlier draft tried to make the two numbers agree — the bid was
worth "exactly what its entry was taken on" — and the claim was false the moment a past neighbor was already
covered. The two are different by design, and the honest statement is that neither ever reads the other's.

**On D12 — why the neuron can compute what a pattern is worth.** It knows what the file pays for an activation with
the pattern in its cover and what it would pay without: both are counts over the activation, and the neuron holds
what every activation gives every pattern it has (D21). The difference between the two is the whole of the benefit,
and it is a fit against the neuron's own evidence. What it never needs is the file — a length nothing computes
cancels out of a difference (R12).

**On D12 — conservative, and in a stated direction.** The line is paid once over the whole run, so a pattern
that pays for itself within `H` of its neuron's own activations pays for itself many times over in the file. The
test asks for the stronger thing. What it therefore drops is structure that still describes the run but has
stopped describing the neuron's recent situation — which is the adaptation D24 is for, not an error in the
estimate.

**On D12 — one baseline, two populations.** Every test asks what the file pays with the neighborhood against
what it pays without. Without it, each neuron it held goes to whatever else names it — another pattern of the
same cover, or another accepted bid one level up — and otherwise into the residual, where it costs the line it
would always have cost. **The flat file is not a second baseline**; it is what that question returns when
nothing else names the neuron.

What is left between the two tests is the population and the line. R12 sums over the `H` activations in the ring
and charges `1 + |e|`; R21 sums over one frame and charges nothing, because the line was already weighed where
the pattern lives.

**On D12 — why `L` never appears in the arithmetic.** `L` is what the differences are differences *in*; it is
the reason the arithmetic is the arithmetic, and it is never a term in it.

**Why a neuron's history states nothing.** It is evidence, not an encoding: it records what was seen so the
collapse can center on it (R4), and a decoder never reads it. The one place a price is needed — does this
pattern earn its dictionary line — is a drop in `L` the neuron works out for itself, over its own evidence, and
nothing crosses between scopes to correct it. What that costs is stated in
[algorithm-evaluation.md](algorithm-evaluation.md): a neuron whose territory another unit reliably takes keeps
pricing its patterns as if it did not.

# 4. Neighborhoods and distance

**On D13 — a pattern is a name for a chunk of spacetime.** One pattern-learning algorithm and one kind of
pattern. There are four types only in the sense that two declarations cross: an **offset** may be zero or
spanning in any activation dimension, which is spatial against temporal, and a **neuron dimension** is event or
action. Neither is a separate mechanism. Spatial is a setting of the reach, and an offset does not know which
kind it is. Event and action are not separate for a plainer reason than it looks — the machine observes its
own actions, since each action dimension carries what was executed, so an action is a symbol read back the way
a pixel is and a pattern over it is learned by the same counting. **You could not tell from a dictionary line
which of the four you were holding.** The one asymmetry lives outside the pattern: an action is chosen and an
event is only expected (R35).

**On D13 — siblings at offset zero.** Several children promoted at one coordinate are activations of different
neurons at the same place, and above the base they see each other at offset zero in every component. That is
what lets the level above chunk them: two patterns of one neuron that keep being bought together are, one
level up, two neighbors that keep co-occurring, and a pattern over the pair is the merge the neuron itself
cannot build (R14).

**On D14 — why the reach doubles.** Each level holds at most half the activations of the one below (T11), so
its units stand twice as far apart, and the box has to double for the expected number of neighbors — a level's
density times the volume of its box — to stand still.

**On D14 — both ends are pinned by what is already given.** The declared alphabet fixes the bottom.
`log₂N₀` levels of halving take `N₀` active neurons down to one, and at that depth the reach expression returns
a reach spanning the whole active region — so nothing has to state that the apex should see everything.

**On D14 — why every dimension grows by the same factor.** The thinning factor comes from a level's activation
count, and a level has one of those, so the schedule cannot tell the axes apart. Growing them all by the same
factor is the isotropic solution and the right default when nothing distinguishes them. Data that chunks harder
along one dimension than another thins faster there and would want that dimension's reach to grow faster;
measuring spacing per dimension instead of one count per level is what that would take, and whether it is
worth it is open ([algorithm-evaluation.md](algorithm-evaluation.md)).

**On D14 — the buffer, walked through.** With a reach of 1 — the base — an activation at frame 10:

```
                 frame 10        frame 11
   buffer        [ 9 10]        [10 11]
                      ▲              ▲
   activation       fires        +1 fires, with its reward
   at frame 10   newest edge    connects, closes
                   sees −1
```

The backward half is read at the newest edge and never again; the buffer's only other job is to be the
frames the next activations read their own backward halves from.

**On D15 — what made coarse offsets safe.** One neighbor per offset was a consequence of atomic offsets,
never a rule. Logarithmic offsets make reach exponential in the alphabet, which is what makes the reaches D14
schedules affordable to write down at all. Neither end of the offset alphabet is told which to use, and
neither declares anything (R4).

**On D15 — how the decay plays out.** A level whose units stand one position apart uses the exact end and
votes the coarse offsets away for want of a majority; a level whose units stand twenty apart does the reverse.

**On D16 — the price is not a notion invented for matching.** It is literally the symbols that would follow
the activation in the file: a neuron named and absent has to be turned off, and that turn-off is the whole of
what a neighborhood is charged. A neuron that fired and nothing named costs one line whether or not the
neighborhood exists, so it is charged to nobody — which is why the residual is a term of the activation and not of
any pattern.

**On D17 — the split is availability, not meaning.** An activation sees both directions, and a pattern is minted
after its chunk has been seen. Covering is simply what cannot wait. This binds action neurons no less than
event ones: an action neuron fires when its action executes and connects forward the same way.

**On D17 — the forward half is the child's, not the parent pattern's.** An earlier draft kept the forward half on the
pattern: the collapse over what followed every activation the pattern covered, read by the child when the child stood
on the apex. Three things argued for moving it to the child's own connections. The child exists in exactly one
situation — its parent's pattern was bought — so its connections are already the distribution of what that situation
was followed by, with no pattern needed to condition it. The child's connections are in its own level's alphabet at
its own reach, so the top of a stack expects at the widest reach the stack has, where reading the parent's pattern
would have it expect at the reach of the level below. And the pattern's population was looser: it summed every
activation where the pattern applied in the parent's own cover, bought or not, and applied-but-not-bought usually
means something else described the chunk better, which is a different situation. What it costs is that a newly minted
child's connections start empty, where the pattern would have carried a future over from before the mint. The cost is
one frame: the child is given an activation at its mint and its connections form from the next frame on (R13), so by
the time it is first bought it already holds what followed the situation it was minted in.

**On D17 — the base speaks its marginal.** A base neuron on the apex has been recognized as nothing more
specific than itself, so its own connections — what follows the symbol over every situation it fires in — are the
best expectation anything has for it in that frame. It is coarse, and it is silenced the moment something
more specific is bought over it (D7). The alternative, the base expecting nothing, left the machine mute and
its exploration stalled until the first pattern was bought, which was a real gap.

**On D17 — siblings.** Two children promoted at one coordinate are two neurons with two sets of connections. While
they are always bought together their activations see the same forward neighbors and their connections agree; the
first time one is bought without the other, they diverge. Where they never diverge, the level above sees them as two
neighbors at offset zero that always co-occur and merges them (§4).

**At `R_t = 1` the vocabulary collapses.** `O⁻` **is** `O` backward, nothing is in flight, and the forward call
delivers one frame. Contraction loses its cross-frame contention, since every bid spans one frame. Read the
document with the temporal parts struck out and it is the spatial algorithm, unchanged — the whole machine, not
a stage of it, which is what makes a reach a configuration rather than an architecture.

**On R1 — nothing waits.** An earlier design held every structural decision open until the forward half had
landed, on the argument that a child names a whole span and half of it had not happened. But the half that had
not happened was never a term in either test. What follows a neighborhood is measured rather than chosen
(D17), so waiting for it bought nothing and cost the entire apparatus of commitments, horizons and provisional
answers that used to sit between the two ages.

**On R2 — prices and structure move at the same moment and are still different kinds of thing.** Both move
when a neuron fires, because that is where counts move and where both tests run (R13). But a price is
re-derived from whatever the table currently says, while a structural move — adding, retiring (R15, R17) — is a
decision that stands until something reverses it.

# 5. State

**On D18 — why the two objects can never be one.** Being a center (T2), a neighborhood is typically a set no
activation ever was.

**On D20 — why the record holds no absolute time.** Expiry was the frame number's only reader and expiry is a
FIFO depth (R9), so the record has nothing absolute in it but the `position`, which the machine supplied when
it called — and which no comparison, price, count or vote ever reads.

**On D20 — why there are no bins.** An earlier design grouped activations by identical backward half and gave the
group one cover, on the grounds that R18 reads the backward half and nothing else, so equal inputs get equal
covers. R6 breaks that: an activation keeps the cover it has unless a re-derived one is strictly cheaper, so two
activations with one backward half can hold different covers depending on what the table was when each was saved.
The group could no longer share, so the group is gone. Nothing was lost but a cache — every sum the tests need
is a sum over activations either way (T3).

**On D20 — why the connections are one set, on the neuron.** The design used to keep forward tallies on the
pattern and connections on the neuron: per-offset counts of what followed, and per-age estimates of what an
action was worth. Both were indexed by a forward offset and a neighbor, and the distance a connection was held
at was exactly the offset the action fired at. So the connection was a forward neighbor that happened to be an
action, carrying one more number. D20 says so: an action connection is a forward neighbor with a strength and an
estimate, the event connections sit beside it, and the set is the neuron's over its own life — which is where
the connections always were.

**On D20 — weights, not sets, and no window.** The backward side needs a window and a majority because it is
written into the file: a pattern has to be a set, since a set is the only thing a decoder can expand, and the
ring is what the majority is taken over. The forward side is never written anywhere (D9), so it needs neither.
What a neuron keeps forward is every neighbor that ever followed it, at every offset, with how often — the
empirical distribution of what follows the symbol — and for an action the sample mean of what it earned. That
is the maximum-likelihood statistic for a thing that is only ever read, and the vote at the base is a mixture
of those distributions with one unit per voter (§13). A majority would throw the minority away for no reason
the file gives, and a window would forget for no reason the file gives.

An earlier draft kept the connections as a majority over the ring, on the argument that a weight that only
ever grows answers a changed world at the rate it can be outgrown. It does; and that is not where the design
answers a changed world. A neuron's connections are the lifetime marginal of what follows its symbol. Specificity
comes from the hierarchy: when the world changes the neuron's patterns turn over within `H` (D24), a new child
is minted, and that child's connections are over the new situation alone from its first exposure. The old estimate
is not wrong; it is the general case, which is what a base neuron is for (R35). Responsiveness is bought with
structure, not with forgetting, and the window stays where the file needs it.

**On D21 — why handover is arithmetic.** What an activation holds against each pattern is the index R6 says nothing
has to be added to — a pattern that moved recomputes what it covers in each of them, and every activation reaching
for it is current again. An activation's share moves whole, so a pattern joining or leaving a cover transfers its
share in `O(offsets)`. The offset grid grows with the level, since D14's reach does, while the number of
neighbors in it stays fixed by construction — that is the invariant the reach is chosen to hold.

# 6. Counts, the collapse, re-centering

**On D23 — why a pattern tallies neighbors it does not name.** A pattern used to count only the neighbors
assigned to it. That is enough to decide whether to *keep* a named slot and never enough to decide whether to
*enter* one: a neighbor the pattern does not name is never assigned to it, so its count was identically zero
and R4 could never take it. R4's abstention paragraph says a pattern grows into the residual, and the state as
defined could not support that sentence; R5's "free, because the counts are already maintained" was not true of
the one count that growth depends on.

The fix is the smallest that makes R4 exact. At each slot R4 wants two numbers: how many covered activations had
the neighbor there and unclaimed by another pattern of the cover, and how many abstain because another pattern
holds it. Those are `present(p)` and `held(p)`, and the population is `n − held(p)`. Both are sparse — indexed
by neighbors actually seen — and both move a whole activation's worth at a time, so R3's granularity is unchanged.
The one new obligation is R3's third line: when an activation's cover changes, a pattern that *stays* in the cover
still subtracts and re-adds, because what another pattern took from the residual moves this pattern's
`present` to `held`, or back.

**Worked case.** `e = {b@0, c@0}` covers ten activations and `d@0` begins to appear. `d` is residual in every
activation that has it, so `present_e(d)` climbs by one per such activation while `held_e(d)` stays zero. At `2 ·
present_e(d)
> 11` — six of ten — re-centering enters `d` and `e` becomes `{b, c, d}`. The four activations without `d` now price
`e` at 2, which is what they were paying before (line plus one residual), so R6 lets them keep it, and they
evict in turn. **No candidate could have done this**: a candidate is built on the residual alone (R14), `b` and
`c` are held by `e` in those activations, and `{d}` alone saves nothing (R15).

> **T2 — The collapse is the per-slot minimizer of the pattern's margin over its population.** Over the
> activations a pattern covers, naming a neighbor moves the summed margin by `+1` wherever that neighbor was in the
> residual — one more neuron covered — by `−1` wherever it did not fire — one more symbol charged (D16) — and by
> `−1` once, for its place in the line (D10). Where another pattern of the same cover already holds it,
> nothing moves at all: no `covers` to gain, no `price` to pay. So the population for that slot is the activations
> of the first two kinds, and the neighbor pays exactly when `2 · count − n − 1 > 0`, which is R4's rule. The
> slots are independent, so the per-slot rule minimizes the sum. It is a *center*, not a medoid: synthesized,
> possibly a set the neuron has never seen. That is the point — it is the typical neighborhood, not a sample
> of one.

**On T2 — why a centroid will not do.** A centroid over sets is a fractional vector, which is not a set, cannot
be written into the file, and has no symmetric difference. The counts **are** the fractional object; the
collapse is how the design gets from it to something the decoder can expand.

**On R4 — why the line is in the slot rule, and why the rule is not `2 · count > n`.** The threshold is a
file-length statement, not a majority statement, and the dictionary line is part of the file (D9). Naming `p`
in a pattern that covers `n` activations, `count(p)` of which have `p` in the residual:

```
history    − count(p)          those residual lines are gone
history    + (n − count(p))    the activations without p now carry a wrong name
dictionary + 1                 the pattern's line is one symbol longer
```

Net change `n − 2 · count(p) + 1`; the slot is taken when that is negative, `2 · count(p) > n + 1`. Dropping it
is the mirror. **The plain majority counts the history and forgets the line**: at three of five, naming saves
one line of history and costs one of dictionary, and the file is the same length. An earlier draft took a slot
at that bare majority and charged the line only in the tests that add and retire (R15, R17), which left the
per-slot decision off the objective by exactly one. Charging it where the slot is decided makes every slot
decision a strict descent on the margin, which is what T6 needs, and it removes the last place a neighborhood
could grow at no gain.

**On R4 — why the slot holds at equality.** With the line charged the boundary is `2 · count = n + 1`, and at
that boundary naming and not naming cost the same. A rule that dropped the slot there could re-add it next
bill when one activation moved and drop it again the bill after, walking a plateau forever without ever lowering
the file. Holding what it had makes a plateau a fixed point, so a pattern that has stopped improving stops
moving.

**On R4 — the third case is why abstention exists here and nowhere else.** An activation is never allowed to sit
out a question the design is asking it; this one it has already answered, for that slot, by having the
neighbor covered.

**On R4 — why there is no forward rule.** The collapse exists to turn counts into a set, and a set is needed
only because the file has to be expanded. A connection is charged nothing: it is not in the line and not in
the history (D9). So there is nothing to break even against and nothing that has to become a set. An event
slot goes down as a vote at its strength (§13), an action connection as an inference at its strength and estimate
(R36), and a majority would throw away exactly the minority the vote still needs and exactly the alternatives
the walk is for (R37).

**On R4 — uniqueness is not assumed and is no longer guaranteed.** At the base, an offset naming one position
holds one neuron of a dimension (D5), so those counts sum to at most `n` and only one can clear the half. Above
the base several units may fire at one coordinate, so several can clear it and the neighborhood names them all
— which is D15's coarse-offset case arriving for a second reason. `|e|` counts them, and nothing else in the
design had to change for it.

**On R5 — three consequences, and they are the point of the design.**

- **Neighborhoods track their demand.** A pattern created on thin evidence is pulled toward its cluster.
- **Coincidence is voted out.** A neighbor present once loses its majority to silence and drops out.
- **Reach emerges.** Offsets where nothing recurs fall away. How far a pattern reaches is discovered, not
  declared. The radius bounds it; it does not set it.

**On R5 — why once per bill and not twice.** An earlier draft re-centered after the activation was saved and again
after the tests that add and retire, because the cover was a partition R18 made fresh every time, and a pattern
joining or leaving moved every other pattern's share. R6 holds covers, so a pattern added this bill joins an
activation's cover only where that is cheaper, and a retired pattern's activations re-derive once. What those
moves do to other patterns' counts is real, and it is re-centered at the next bill. Re-centering it now would
decide the center on the order the two moves happened to run in, which nothing else in the design is allowed
to read.

**On R6 — why covers are held.** R18 is greedy, and a greedy cover re-derived after a pattern moved can cost
more than the one that stood. In Lloyd's algorithm the assignment step is exact, so re-assigning after the
centers move can only help. Here it cannot be exact — an exact cover is set cover — so the design keeps the
old assignment unless the new one is strictly better. That is one comparison over numbers the neuron already
holds, and it is the difference between a bill that descends `L` and one that can raise it (T6).

**On R6 — why nothing is indexed the other way.** What each activation holds against each pattern is already the
index (D21), so a reverse map from pattern to the activations it covers would be a second copy of the same fact.

**On §6.2 — why a set and not a distribution.** Covering needs a set. So does the file: every slot it states
holds one symbol or nothing.

# 7. The history

**On D24 — one count, so one denominator.** What R12 needs is not a shared clock but a shared divisor: a
pattern's benefit is a sum over the activations it covers, and for two neurons' tests to mean the same thing that
sum must be over comparably much evidence. A uniform `H` gives that directly. A shared *window* gave it only
for neurons firing at similar rates, and gave a rare neuron almost nothing to decide on.

**On D24 — this is adaptation, not forgetting.** A neuron that keeps firing sheds its old activations as new ones
arrive, so patterns describing a situation that has passed stop being taken into covers, drain their benefit
and are retired (R17). A neuron that falls silent sheds nothing: it holds its `H` activations and its patterns
intact, indefinitely, and resumes from them when its situation returns. An active neuron adapts exactly as
fast as its evidence turns over, and a silent one simply waits.

**On D24 — `H` does three jobs.** It is the structural memory — connections are outside it (R31) — it
is R12's selectivity — double it and every pattern's
benefit roughly doubles against an unchanged line, so more survive — and it is the rate at which the stack
deepens (T13). One number, three effects, all monotone in it, and it should be tuned knowing that.

**On R7 — duplicates die in the table, not the market.** Two patterns with one neighborhood are taken into a
cover older first, so the younger holds nothing anywhere and retires. The market would kill the younger too —
the election ties to the older symbol (R23) — but the neuron never hears the election's verdict, so the table
has to be able to do it alone, and it can.

> **T3 — Counts are sums over activations, and nothing reads an activation whole but eviction.** Re-centering sums
> `present` and `held` over the activations a pattern covers (D23), both tests sum margins over activations, and the
> candidate collapses over a population of activations — every one of these is a per-slot count, and the per-slot
> counts are what a pattern keeps (D23). Forward, there are no activations to sum over at all: a connection is
> a total on the neuron, strengthened as neighbors fire and never read back per activation (R31). Nothing asks
> whether `c` at `+1` came with `d` at `+2`. The one operation that needs an activation as a unit is removing it
> (R8), because a sum cannot say which of its terms was the oldest.

> **T4 — What covers is what was priced.** An earlier design chose a cover on the backward half and then priced
> the activation on the whole span, so the pattern that won the prefix could end up a worse describer than one that
> lost it, and no pass could reconcile them.
>
> There is nothing left to reconcile. The cover is chosen on `O⁻` and priced on `O⁻` (D17), so the pattern that
> took a neuron is the pattern charged for it, at every reading and for the life of the activation. **The forward
> half is never a term in that comparison**, so it cannot contradict it.

**On R8 — why records and not a summary.** A total cannot answer retirement: when a pattern goes, its neighbors
have to be re-covered from the table, which needs the activations and what each holds against each pattern — a
single number per pattern could not produce it. Keeping the ring is not a storage saving; it buys that both
tests scan distinct backward contexts and read pre-summed counts.

> **T5 — Every loop in the design is bounded, and none is capped.** A bill is five passes, each over a fixed
> set — the ring, the table, the slots — and none of them repeats until a condition holds (R19): one candidate
> is built by one seed and one collapse, one pattern retires at most. The election is two decisions and a
> settling, none of them repeated (R23). A retirement is collected within `reach_t(D)` frames and takes
> its whole subtree in one step (R17). The level stack is bounded by the base activity behind a frame (T12) and
> by the run so far against `H` (T13), which together also bound the settlement walk (R24) and the depth of an
> expansion (R27). Every bound falls out of a quantity the design already counts, so nothing has to be chosen
> to make the machine halt.

**On R11 — the trade, stated plainly.** Structure is not dropped for being old; it is dropped for having
stopped paying against the last `H` things its neuron saw. A neuron in a changed situation restructures at the
pace of its own new evidence, and one whose situation has simply gone quiet keeps what it had. Absence of
evidence is no longer read as evidence the structure died.

**On R10 — why the alphabet is not a parameter.** A resolution defines what a base symbol *is*, so it states
the problem rather than tuning the algorithm. Everything about depth is derived: adjacency is the reach read
conjunctively, and the reach per level comes out of one expression.

**On R11 — there is no floor relating `H` to the reach.** There is no window for a span to be wider than: the
file is the run (D9), so every symbol is priceable at any reach.

# 8. The one test

**On R12 — the tests a symbol passes through, in one place.** The line brackets the symbol's life and the
elections fill in the middle. Every row is stated by the rule it cites; this is a reading aid, not a rule.

```
build     would what C takes out of the residual sum past 1 + |C|?      the line, prospectively   (R15)
cover     does this PATTERN take more of the residual than it costs?    one activation, no line       (R18)
offer     does more than half of this PATTERN fire?                     one activation, no price      (R18)
elect     does this BID cover more than it costs, once slots are split? one bid, no line          (R23)
retire    does what e still keeps out of the residual pass 1 + |e|?     the line, retrospectively (R17)
```

**Cover and elect are one expression** (D12) — what a neighborhood covers against what it costs to state —
asked over one activation and over the board. Build and retire are that same expression summed over the history
with the dictionary line added, read in opposite directions: what an absent pattern would take out of the
residual, and what a present one is still keeping out of it. The offer is the one row that is not a price, and
§10 says why.

**On R12 — a benefit can be zero for two different reasons**, and both are the signal. Zero because the
neighbors are already covered by another pattern of the same cover — no sharper child here would shorten the
file. Zero because the next pattern in line fits the activation just as well — the pattern duplicates something the
table already holds. A pattern accumulating either drags itself toward retirement, and neither needs a
mechanism aimed at it.

**On R12 — the movement of benefit is cheap.** A pattern gaining or losing an activation and an activation joining or
being evicted are `O(offsets)` off the counts; a re-center is the walk the scan is already making (R19).

**On R12 — a newborn needs exactly the bracket and nothing more.** Where its territory was residual, the slots
are free and it is bought on its first recurrence — no line at the election means no deadlock at birth. Where
its territory turns out to be another neuron's, the election declines it and the neuron never learns why; the
pattern stays as long as it pays on the neuron's own books, which is the trade
[algorithm-evaluation.md](algorithm-evaluation.md) records.

> **T6 — On a fixed history, the bill descends the neuron's file and stops.** Write the neuron's file over its
> ring as
> ```
> L_N  =  Σ over activations f  [ |residual(f)|  +  Σ over the cover of f ( 1 + |e⁻ \ f⁻| ) ]
>      +  Σ over patterns  ( 1 + |e| )
> ```
> the uncovered neurons, a line and its charges per covering pattern per activation, and the dictionary — D12 read
> over one neuron's evidence. Freeze the ring. Then each pass of R19 is non-increasing on `L_N`:
>
> - **Build.** Adding `C` changes `L_N` by exactly the negative of R15's margin: over the activations whose cover
>   `C` joins it removes what it takes from the residual and adds its line and its charges, and it adds one
>   dictionary line. `C` is added only when that margin is strictly positive, so `L_N` falls by at least one.
> - **Retire.** Removing `e` changes `L_N` by exactly R17's margin, with the sign reversed: its neighbors that
>   no other pattern of the cover names return to the residual, its lines and charges leave, its dictionary
>   line leaves. `e` is retired only when that is strictly negative, so `L_N` falls by at least one.
> - **Re-center.** For a fixed cover and assignment, `L_N` is a sum over slots of independent terms, and R4's
>   rule takes each slot exactly when its term falls (T2), holding at equality. So a re-center is non-increasing
>   and strictly decreasing whenever a slot changes.
> - **Re-derive covers.** R6 replaces an activation's cover only by a strictly cheaper one, which is the cost of
>   that activation in `L_N` falling. Nothing else moves.
>
> `L_N` is a non-negative integer, so the strict moves are finite, and the process reaches a state where no
> candidate pays, no pattern is negative, no slot moves and no cover is cheaper. That is a local optimum with
> respect to exactly the moves the neuron has. On a sliding history it tracks one, which is all that can be
> asked.
>
> **What the theorem rests on, and what happens without it.** Two things. The partition: with each present
> neighbor credited to one pattern of the cover, `L_N` decomposes over patterns and the re-center is a descent
> step. Without it a pattern's own fit and the file disagree wherever two patterns name one neuron — a
> candidate born on unmet ground can re-center onto ground another pattern holds, look better and better to
> itself, be worth less and less to the file, be retired, and be rebuilt from the same residual by the same
> seed. On a frozen history that cycles forever. And the hold on covers (R6): without it the greedy cover can
> cost more after a re-center than before, and `L_N` can rise. One more detail sits inside the hold: when a
> candidate is installed, an activation must be offered its held cover with the newcomer appended, not only the held
> cover and a fresh re-derivation, because the appended cover is what R15 priced and a fresh re-derivation can
> land between the two — cheaper than what stood, dearer than what was counted, with the dictionary line
> charged in full. With all three, the descent is monotone.
>
> **What it does not say.** Not that the optimum is global — choosing the pattern set is set cover, and the
> concrete local optimum is a history where `a, b, c, d` always fire together held by `{a, b}` and `{c, d}`,
> each paying, neither retirable, the merge never proposed because nothing is ever unmet (R14). That merge is
> the level above's (§4). And not that `L`, the file over the run, descends: `L_N` is one neuron's reading of
> it, and what the election does with the neuron's patterns is not in `L_N` at all.

---

# 9. The two moves

**On R13 — why a mint activation.** A pattern is minted at the parent's age 0, on a backward half that is
whole (D17), and everything that follows that frame is exactly the future of the situation the child was
minted for. Had the child waited for its first purchase to open an activation, that future would have gone by
unrecorded, and the child would begin its life one situation behind. The activation costs nothing structural:
it is not elected, so the file, the frontier and the vote are untouched, and the only thing it does is what
every activation does between activations — strengthen its neuron's connections (T8).

**On R14 — building a candidate, worked through.** Five activations in the ring, an empty table, so every
neuron of every activation is in the residual.

```
o₁⁻ = {a,b,c}   o₂⁻ = {a,b,d}   o₃⁻ = {a,b,c}   o₄⁻ = {a,b,e}   o₅⁻ = {x,y}

seed        a and b are each in four residuals; a is earlier in declaration order, so a
population  o₁ … o₄, the activations whose residual holds a       n = 4,  2·count > 5 to name

collapse    a: 4 → 8 > 5  named      b: 4 → named      c: 2 → 4 > 5?  no
            d: 1  no      e: 1  no

C⁻ = {a,b}      reach   2, 2, 2, 2  over o₁…o₄;  o₅ names nothing C holds, so R18 would not take it
                saving  1, 1, 1, 1, 0        benefit 4  >  line 1 + 2 = 3     requested
```

By hand. `o₁` used to pay three lines. With `C` in its cover it pays `1` for `C`, which names both `a` and `b`
and gets neither wrong, plus one line for `c` in the residual — two symbols instead of three. So do `o₂`, `o₃`
and `o₄`. `o₅` shares nothing with `C`: taking it would cost `1 + |{a,b}| = 3` against two neurons it does not
even name, so R18 never puts `C` in that cover and `o₅` pays its two lines exactly as before.

**On R14 — what the loop was doing, and why a seed does it in one step.** The earlier build grew `C` a
neighbor at a time, taking the largest net gain each round and stopping when none paid. Every round was a
majority in disguise — the neighbors in the residual against the neighbors absent, over the whole ring — but
over a population that changed as `C` grew, which is the only reason it took a round to add one neighbor.
Fixing the population first removes the rounds. The seed is the neighbor the table is failing on most; the
population is the activations where it is failing; the collapse over that population settles every other slot at
once by exactly the majority the loop was computing. The seed chooses the population and the population
decides every slot. Nothing in either place grows anything.

**On R14 — what "the same history" means.** Covers are held rather than derived (R6), so two neurons with
identical rings can carry different covers if their tables moved under them in a different order, and the
residual — and so the seed — is a function of the ring and its covers together. The build is deterministic in
that pair, which is what a fixed-pass construction can promise; it is not a function of the ring alone, and
an earlier draft said it was.

**On R14 — why the candidate is not one of the activations.** A neighbor enters `C` only while more of the
population hold it than not, so what a single activation carried alone never gets in — and over a span
`reach_t + 1` frames wide a single activation carries every coincidence in the window. Minting one raw would charge
a line for those coincidences and then re-center them away at the next bill.

**On R14 — why nothing stands in front of building a candidate.** A gate would have to be a threshold on how badly
something was being covered, and the design settles nothing on a count of mismatched neurons. It is also
unnecessary: where the table already describes its activations well, the residual is thin, the seed's population
is small, the collapse over it names little, and the price refuses it (R15). **The signal goes quiet by
itself** once every pattern's slots sit near `0` or near `n`, which is exactly when there is no work left.

**On R14 — what a candidate costs to build.** One tally over the ring for the seed — how many residuals hold
each neighbor — and one collapse over the seed's population, per slot. Both are the walk a re-center makes.
The whole construction is `O(H · w̄)` with `w̄` the neighbors an activation holds, and it is the reach that sets
`w̄` (D14).

**This is facility location.** Activations are customers, patterns are facilities, opening one costs `1 + |e|`,
serving costs what the neighborhood names wrongly, and the cover pass is the assignment. The opening cost is
the only thing standing between the design and memorizing every frame: if opening were free you would put a
warehouse on every customer. The local search is usually given four moves; here **split**, **merge** and
**swap** need no machinery of their own. Split is what R14 does — a candidate takes the part of a pattern's
demand that shares a seed — merge is what retiring does to redundant patterns, and swap is a candidate built
at one bill and a pattern retired at the next — the child takes the activations, and the pattern it stranded fails
R17 at the next one.

**On R15 — the test asks the question R18 will answer.** It prices `C` on the residual of the activations in the
ring, and R18 will take `C` into a cover on the residual of an activation — the same quantity, over the same
evidence. **There is no bet left for R17 to collect on**, and nothing can hand `C` less than the
test counted except the history moving on, which is R17's ordinary business.

**On R15 — a candidate rejected today is not lost.** Every later bill builds one again, over a residual the
saving and the eviction have moved, so what does not pay for its line today is minted as soon as the activations
behind it recur enough to pay for it. Re-centering then means a child that does get minted improves with
exposure rather than freezing at the shape it was cut to.

**On R15 — one per bill is not a limit on how much structure a neuron can build.** A neuron fires once per
frame per position, and every activation is a bill. What one bill leaves unmet is the next bill's seed. A neuron
that needs three patterns builds them over three of its own activations, which is the same rhythm the machine
keeps: one election per frame, and the level above built from what it bought.

**On R16 — what makes release safe** is R17's condition rather than any wait: a pattern is deleted only when
its child has nothing open, so the neuron released has no open activations, and by the same argument neither
does anything beneath it. R13's "never pays off on the evidence that created it" is about the neuron; the
pattern covering that activation is not the same claim.

**On R17 — why one and not every negative margin.** Two patterns straddling one cluster are each worth nothing
while the other stands: whichever is removed, the other picks up its neighbors for free, so each margin reads
as if the other were doing the work. A pass that retired both on one reading would return the whole cluster to
the residual with nothing left to cover it, and the next bill would rebuild one of them. Retiring the worst
alone lets the survivor's margin, read next bill, carry the whole cluster. The earlier design did the same
thing inside one bill by re-checking after each retirement; doing it across bills is the same sequence with no
loop.

**On R17 — a candidate cannot be retired by the pass that follows it.** After a candidate is added, what R15 priced
and what R17 reads are the same set counted the same way — the residual `C` took, measured against
the same table. The margin R17 reads is the one R15 just found strictly positive, and one
retirement can only hand `C` more neurons or remove a competitor. A pattern only ever falls below its line by
losing neurons to another pattern of its cover or by having its activations evicted.

**On R17 — what two sequential tests cannot reach.** A candidate that would pay *only* if some incumbent's line
were refunded fails R15 and is never put to R17 — two patterns straddling one cluster, each carrying its
weight while the other stands, neither individually deletable. Pricing that case would need a third move with
a formula of its own, joint over adding one pattern and retiring another. It is not worth one. Re-centering
pulls an off-center pattern to its cluster without being asked, and a straddling pair survives only until drift
or eviction starves one of them. **The miss is in the safe direction**: what a greedy build-then-retire gives
up is a compression not taken, where a candidate priced against the flat file gives up a file made longer.

**On R17 — why the subtree needs no cascade.** A retired pattern's child cannot fire again, so by the death
frame it has no open activations; if it has not fired, none of its children has fired either, so none of them
has open activations, and so on to the bottom. **Children outlive their parents** — reach grows with the
level, so an activation above is still open when the one that fed it has closed — and the death frame waits on
the child's last activation for exactly that reason.

**On R17 — nothing irreplaceable dies.** A pattern retired while its evidence is still in the ring is rebuilt by
R14 the moment that evidence pays again.

**On R17 — why the death ledger needs no back-pointers.** A back-pointer from a child to whatever is naming it
would be structure the file does not hold, and the machine already holds the open activations that answer the
question.

# 10. The frame, per neuron

**On R18 — why the cover is a set and the criterion is a ratio.** A neuron picking one pattern has a nearest
neighbor problem; a neuron picking several has a covering problem, and covering is where a ratio belongs. A
line is paid once per pattern however many neurons it accounts for, so what matters is not which pattern is
closest but which buys the most residual per line. **That is not merely the same criterion R23 uses one level
up; it is the same procedure**, run by the neuron over its table and by the machine over a frame's bids. What
differs is the population and the fact that only the neuron may mint or retire a symbol.

**On R18 — why the offer is wider than the cover.** The cover is the neuron's own partition, chosen on the
neuron's residual. Take an activation `a, b, c, d, e` with patterns `E1` naming `a, b, c, d` and `E2` naming
`c, d, e`. The cover takes `E1` first and leaves `E2` with `e` alone, one neuron against a price of one, so
`E2` is not in the cover. Now let the machine already hold `a` and `b` from a past frame's unit. `E1` is worth
`c, d` less its line, one; `E2` is worth `c, d, e` less its line, two; and `E2` was the better purchase. An
offer restricted to the cover never shows it to the machine. The offer is therefore every pattern that
applies, and the machine ranks.

**On R18 — why the apply test is a majority and not a price.** The offer needs a filter that drops nothing
the machine could buy and sends nothing it could not. A pattern whose present neighbors do not outnumber its
absent ones has `covers ≤ price − 1` on any board, since the board can only take present neighbors away, so it
can never clear R23 and there is no reason to send it. A pattern whose present neighbors do outnumber its
absent ones might clear, depending on what the board has already paid for, and the neuron cannot know. So the
loosest safe filter is exactly the majority, and it is the collapse read backwards: a pattern is a majority
statement over the activations it covers, and an activation is one the pattern describes when it agrees with the
majority of the statement. The price belongs to the buyer.

**On R18 — why the cover keeps its own test.** The cover is not an offer; it is where the neuron's counts come
from. Taking a pattern into a cover on a bare majority would credit it neighbors it does not pay for on the
neuron's own books, and T6 needs the neuron's books to be the file's. So the cover keeps `covers > price` and
the offer takes the majority, and the two sets differ exactly where the neuron's residual and the machine's
would.

**On R18 — why nothing comes back.** The neuron has already decided everything, on a whole backward half, and
the election settles who the machine paid. An earlier design reported the election back as a fact per
neighbor — which of this activation's neighbors another bid was credited — so the neuron could price its
patterns on what it actually sold. The report is gone because the accrual it needed was the most involved
machinery in the design and the case it guarded against was judged rare: a neuron consistently outbid on the
same ground. What replaces it is nothing. The neuron prices on what it saw, and a pattern that sells poorly
stays as long as it describes the neuron's own activations.

**On §10.2 — why the bill runs before the offer.** The bill used to follow the election, because it read what
the election had credited. With nothing to read, the only reason to split the call is gone, and the natural
order is the one that lets this frame's activation count before this frame's offer is made: save, restructure,
offer. The candidate requested in the call is not offered in it (R13), so the order changes what the table
holds and never what the election sees early.

**On §10.2 — why the bill's decisions are once, not once per activation.** Deciding per activation would impose
an order on activations that are simultaneous — the pixel at one position did not happen before the pixel at
another — and the structure that came out would depend on it, which is the defect R23 removes one level up.

**On §10.3 — why the forward call carries no decision.** Its jobs are transcription: arriving neighbors into the
neuron's connections, a reward into the estimate of the action connection it paid for. Neither feeds a test
that is waiting. What the call does carry out is speech — what the neuron on the apex expects and infers — and
speech reads the connections without touching them.

> **T7 — The tests are the assignment.** The only consumer of a table-wide picture of covers is R14's
> residual, and R17 reads it only through the activations a pattern covers. Both scan the whole
> table anyway. Everything else wants one activation's cover: the cover pass computes it for the activation in hand
> (R18), eviction reads it per departing activation. **So no pass exists to keep a global assignment current, and
> none is needed** — the scan that prices a move is the scan that makes it.

> **T8 — Nothing between activations is read.** Counts move when an activation is saved, when one is evicted, or
> when a cover changes (R3), and all three happen in the frame the neuron fires. Between activations the forward
> call does write — forward neighbors and rewards strengthen the neuron's connections (D6) — but **nothing prices
> them, ever**: no neighborhood, cost or cover reads a connection, and nothing is recomputed in between. The
> connections are read between activations, by the apex, and reading them moves nothing.
>
> **So connecting is evidence in escrow.** It changes what the next answer will be and never an answer already
> given.

**This is Lloyd's algorithm, interleaved with the data.** Assign points to the nearest center, move each center
to the minimizer over its assigned points: Lloyd 1957, better known as k-means. This is its variant over sets
with the file as the distance, the collapse as the minimizer (T2), and `k` moving as build and retire change
the pattern count — which is why those moves exist alongside it, since Lloyd only optimizes assignment for a
given set of centers. Where it departs from Lloyd is that the assignment step is not exact and so is held
rather than redone (R6), and that is what T6 turns on.

**What the design does not do is alternate to stability.** A bill absorbs its evidence, makes at most one
structural decision of each kind and re-centers once: one improvement step, not a fixed point. Iterating would
settle the table against counts the next bill moves anyway, and every bill moves them. The table is never
optimal over the ring and does not need to be. It needs to be current for the next cover, and that is one
activation's costs.

## One activation, across its frames

Take `R_t = 3` and a neuron whose table holds two patterns, `K` and `M`:

```
K names  {(a,−2), (b,−1)}
M names  {(g,−1), (h,0)}
the neuron's connections so far:  an event connection (c,+1), and an action connection (u,+1) at estimate 0
```

**Frame 10 — the neuron fires, and everything is decided.** Its backward half is `{(a,−2), (b,−1), (g,−1),
(z,0)}` — whole, because backward is what an activation already has. The bill runs first.

The cover pass runs over the residual, which starts as all four neighbors. The neuron's own activation is not in
`O⁻` and so is not a slot the cover can take — that is the one place the neuron's population differs from the
board's, where the bidding activation is the first slot a bought bid subsumes (R21).

```
K⁻ = {a,b}   covers 2 of the residual                     price 1 + |{}|  = 1     ratio 2
M⁻ = {g,h}   covers g;  h did not fire                    price 1 + |{h}| = 2     ratio 0.5
```

`K` goes first and takes `a` and `b`. On the second round `M` is re-measured against what is left: it still
covers only `g`, at a price of 2, so it does not pay and is not taken. **The cover is `{K}`**, `g` and `z` are
the residual, and `K` counts `a` and `b` as its own share and `g` and `z` as `present` it does not yet name
(D23).

Then the rest of the bill: the activation joins the ring and the oldest leaves; `K` re-centers; a candidate is
seeded on the neighbor most often in the residual — `g` and `z` are in it this time — and priced; the worst
margin is read and retired if negative.

Then the offer. `K` applies: both of its neighbors are present. `M` applies too: one of its two neighbors is
present, and `2 · 1 > 2` fails — so `M` does not apply, and the neuron returns **one bid**, `K`'s neighborhood
and `K`'s child. Had `M` named `g` alone, it would have applied and been offered beside `K`, cover or no cover.

**The election runs.** Say a neighbor's accepted bid takes `a`, and `K`'s bid is bought on `b` and the bidder.
The neuron is told none of this. `K`'s child is promoted at frame 10 and expands to `a` and `b` both.

**That is the whole of the neuron's frame.** Nothing is held open, nothing is committed for later, nothing will
be asked again about frame 10.

**Frames 11 through 13 — the forward call.** At 11, `c` does not come; `e` fires in that dimension instead,
and the action `u` runs. The machine calls the activation at age 1 with `(e, +1)` and `(u, +1)`, and the neuron
strengthens its connections at `(e, +1)` and `(u, +1)`, creating either that does not exist at strength 1. This
neuron is covered — `K`'s child was bought — so it connects and does not speak. `K`'s child, on the apex at
level 1, is called at its own age with the level-1 neighbors that fired, strengthens its own connections to
them, and returns what its own connections expect and infer at `+2`. The reward for `u` arrives with frame 11
and folds into the estimate at `(u, +1)` on both neurons in the same write (R29). At 12, `(?, +2)` connects the
same way. At 13 the base activation closes.

**The neuron is not asked again, and nothing about frame 10 is revisited.** `K` was not wrong to be in the
cover: it was priced on what fired beside the neuron, and `e` arriving instead of `c` is not a charge against
it (D17). What `e` does is raise the neuron's strength at `(e, +1)` by one against `(c, +1)`, so the next time
this neuron is on the apex its one unit of vote at `+1` is split between `c` and `e` a little more toward `e`
than before (§13). **That is reach emerging** — and it arrives as evidence for the next expansion, never as a
verdict on the last one.

```
   frame 10                                  frames 11 … 14
   ────────────────────────────────────────  ───────────────────────────────
   fire — the backward half is whole         (e,+1), (u,+1) land, then +2, +3
   cover, save, evict, re-center             connections strengthened
   build one, retire one
   offer every pattern that applies          the reward for u lands beside it
   ── the level elects, and says nothing ──
                                             the apex child expects and infers
                                             THE NEURON DECIDES NOTHING
   ────────────────────────────────────────  ───────────────────────────────
   EVERYTHING IS DECIDED HERE ───────────────── evidence for next time ────▶
```

## One frame, as a diagram

The order §3 states, drawn. Every node names where it is specified.

```mermaid
flowchart TD
    A["THE MACHINE holds every open activation, one per<br/>(neuron, age, position), and calls each neuron once<br/>in the frame it fires — §10"]
    A --> B["THE BILL — age 0<br/>cover and save, evict, re-center once — R19 steps 1–2"]
    B --> M["BUILD ONE candidate<br/>seed, population, collapse, price — R19 step 3"]
    M --> DEL["RETIRE ONE<br/>the worst margin, if strictly negative — R19 step 4"]
    DEL --> P["OFFER, and one request<br/>a bid for every pattern that applies — R19 step 5"]
    P -.->|"bids: child id + backward neighborhood"| X["THE ELECTION<br/>take bids by covers per line, credited the free slots<br/>they name, until the best left does not pay — R23"]
    X --> O["THE NEXT LEVEL UP, built out of what the election<br/>bought, at the reach D14 gives it — §12"]
    O --> Z["LEDGER PASS, after the last level has run<br/>delete everything due, subtree and all — §9.2"]
    Z --> W["PROCESS ACTIONS, every open activation at its own age<br/>forward neighbors and rewards in; from the apex,<br/>expectations and inferences out — §10.3"]
    W --> Y["PREDICT — expand the apex's expectations to base symbols,<br/>one winner per event dimension by share of voters — §13"]
    Y --> S["SELECT — expand the inferences to base actions,<br/>one winner per action dimension by estimate; it executes at f+1 — §16"]
```

# 11. Contraction

**On the objective — the machine executes it, it does not evaluate it.** The file over one frame is the units
promoted plus what they got wrong. An earlier draft stated that as a rule of its own — accept a subset `S`,
minimize `cost(S)` — and named it prize-collecting set cover, which is the right classification and the wrong
picture: it reads as though something somewhere forms subsets and scores them, and the classification is only
interesting if you are choosing a search. Nothing searches. R23 takes the bid with the best ratio over the
free set, credits it what it names there, and asks the question again over what is left. **What the objective
describes is the outcome of that procedure, not an instruction to anyone**, which is why it belongs here and
not in the spec.

**On R20 — why the bid carries no forward half.** Nothing at `Δt > 0` has fired, so the machine could settle
nothing against it; the file holds no line for it, so nothing would be priced on it; and it would make the bid
a claim about a frame nobody has seen, which is what the assertion was and what §13 says was removed. A bid is
a dictionary line and a name, and a dictionary line is backward.

**On R21 — why a named neuron another unit covers is free in both directions.** It fired, so it is not among
the neurons named and absent, and no assignment can put it there. It is already paid for, so it is not among
the neurons this bid saves. Zero on both sides, and the two zeros are independent: coverage moves credit and
nothing else. The one case that looks as if it should be different — this bid names the slot *wrongly* while
the other unit names it rightly — is not different, because a decoder expanding this unit still turns on the
wrong symbol and needs it turned off. Being right somewhere else does not make being wrong here free.

**On R21 — with the line in this price, promotion would be impossible outright.** A cover is at most the named
backward neighbors plus the bidder, so it never exceeds `|e| + 1`; a price carrying the line starts at
`1 + |e|`. `cover > price` could then never hold — not on a perfect match with nothing contested, let alone
under overlap. A test that asks one bid to pay an aggregate charge declines every bid.

**On R21 — there are not two clocks to reconcile.** R12 optimizes the dictionary against a neuron's own
history; this price optimizes one bid against the board's coverage. They answer different questions over
different evidence, and neither needs the other's span. What R12 does need is a denominator, and `H` is that
directly (D24).

**On R22 — what earlier-bidder priority costs, and why re-electing the past would cost more.** A bid at `f`
that wins a neuron at `f − 2` keeps it, and a better bid at `f + 1` for the same neuron is credited nothing
for it. The alternative is letting the later bid take it back, which re-scores an election that has already
promoted a unit, and the unit is already a neighbor at the level above. Every re-score would ripple upward. The
bias is real and it is stated; the cases where it bites are boundary neurons between two chunks, and the bid
that loses one counts one fewer.

> **T9 — Coverage settles at `g + reach_t`, and nothing has to wait for it.** An activation firing at `g` names
> backward neighbors across `[g − reach_t, g]`. A neighbor at `f` can be covered by a bid firing anywhere in
> `[f, f + reach_t]`, so the last bid that can touch any of them fires at `g + reach_t` — and the neuron itself,
> at `g`, is coverable until exactly the same frame. Every bid in the argument is at this neuron's own level,
> so one `reach_t` governs throughout.
>
> Coverage is acquired and never revoked (R26), so the frontier over a frame only shrinks. **Nothing in the
> design needs it at a particular frame**: the neuron's bill has already run, and what coverage decides is who
> speaks for the frame — in the file, in the prediction and in selection — which is read live at every frame
> from the coverage set as it stands.
>
> It is also the last frame at which the coverage set holds the activation. The set spans `reach_t + 1` frames
> and ages with the clock, so at `g + reach_t` it holds `[g, g + reach_t]` — the activation sits on its oldest
> frame. One frame later it is gone.

## The election

**On R23 — why the election is R18, and what the earlier rule got wrong.** An earlier R23 resolved every slot
once, on each bid's ratio over the whole free board, and then accepted each bid on what that resolution left
it. That is not the same procedure as R18, though the specification said it was: R18 re-measures after every
take and lets a bid that has fallen below its price take nothing, while the earlier R23 let a bid take slots on
its original ratio, fail its own test, and keep those slots away from the bids that failed *because* of it.

**Counterexample.** Three bids, every named slot fired:

```
A  names s1 s2 s3 s4      price 1   ratio 4
D  names s1 s2 s9         price 1   ratio 3
E  names s9 s10           price 1   ratio 2
```

Old R23: `s1, s2` to A; `s9` to D, since 3 > 2; `s10` to E. A holds 4 > 1 and is accepted. D holds 1, not > 1,
rejected. E holds 1, rejected. The old step 3 moved slots only among accepted bids, so `s9` and `s10` stood as
residual: history term `1 + 2 = 3`. R18 over the board: take A; the free set is `{s9, s10}`; D covers 1 at price
1 and E covers 2 at price 1, so take E. History term `1 + 1 = 2`.

**The frozen ratio is the whole defect.** D's claim on `s9` was worth something only if D was going to pay, and
whether D would pay was not known until after the claim had been honored. Re-measuring per round asks the two
questions in the right order.

**Why the loop is not the variable loop the design avoids.** Every accepted bid subsumes at least two free
slots (`covers > price ≥ 1`), so the rounds are bounded both by half the free set and by the number of bids.
And because `price` is fixed by the frame while `covers` can only fall as the free set shrinks, a bid's ratio
is monotone non-increasing through the election: the top bid can be re-measured alone, and if it still leads
the others' stale ratios it is the true maximum. **The election is a heap pop with one re-measure per round,
not a re-scan.**

**What the rewrite does not touch.** R22's priority for earlier frames — the greedy runs over the free set,
which already excludes everything an earlier election credited. D26's board. R26's frontier. And the fact that
the election delivers nothing to any neuron.

**On R23 — why nothing has to be handed back.** A promoted unit's neighborhood *is* its dictionary line (R20),
so expanding it recovers every neighbor it names, credited or not: coverage is a fact about what the accepted
units expand to, and the assignment has no power over it. The old pass needed a third step to make the
bookkeeping match that, because a slot held by a rejected bid would otherwise read as uncovered and stand as
its own line beside a unit that already delivers it. **The greedy never creates that state** — a bid that never
reached the top holds nothing, so a slot it named is either credited to a bid that did pay or was never claimed
by one at all.

**On R23 — a ratio that ranks is not a price.** `cover / price` is a selection score over two counts: it
estimates nothing, prices nothing, and no cost anywhere is set by it, so T1 is untouched. Both are counts and a
price is at least 1, so the score is a ratio of positive integers and the winner is simply the largest.

**On R23 — why the coordinate tie-break is not decoration.** Two activations of one neuron bid with one
creation order, and in a solid region that is the common case, so without it the pass would have nothing left
to decide with.

**On R23 — why the election says nothing back.** It could report, per bid, bought or not, and per neighbor,
credited or not; an earlier design did the second. The first would let a neuron retire a pattern that never
sells; the second would let it stop naming neighbors it never gets paid for. Both are feedback from the
customer to the business, and both were dropped on one judgment: a neuron finds itself in many situations, wins
some and loses some, and a pattern consistently outbid on the same ground is not expected to be common. If it
is, [algorithm-evaluation.md](algorithm-evaluation.md) says what to measure.

**What is given up.** Greedy is the classical approximation for weighted set cover, and its slack against the
optimum is bounded — `H(n)`, where `n` is the largest neighborhood offered (§17). What it does not give is the
optimum itself, and the case it loses is a boundary one: two chunks sharing a boundary neuron is how a stream
tiles, and the bid that loses that neuron simply counts one fewer. Accepted deliberately, and cheaply —
**contraction mints nothing that lasts**, so a marginal cover costs a bounded handful of lines and nothing
structural.

> **T10 — What settles is a slot, and the settled ones are not a prefix.** Each slot settles once and stays
> settled: settlement is the absence of any live unit that could reach it, and units only ever fall out of
> reach. But the depth over one region is data, so a slot whose stack stayed shallow settles while a deeper
> predecessor is still open. **The edge is ragged and does not sweep.**
>
> **It is ragged in space as well as in time**, since reach is bounded in every activation dimension: a slot
> far from anything active settles while one in a busy region, at the same frame, is still open.
>
> Nothing depends on the order, because nothing is streamed. The file is re-derived whole from current
> structure (D9), so an unsettled frame is not a gap — it has an encoding like any other, one still liable to
> change. "Settled" says what can no longer move, never what has been emitted.

**What settlement settles is the election, not the file.** It marks the point past which no further bid can
reach a frame. What the winning units then *expand to* is the current dictionary's business (D9):
neighborhoods keep re-centering underneath them, and a unit whose pattern has been deleted stops being
available at all, so the run is re-encoded from what remains. **Contraction settles who covers what; D9
settles what that costs to say.**

**On R24 — since the reach grows with the level (D14), the levels that settle last are also the ones that reach
furthest.** A bid accepted at `g + reach_t` can add a level after the fact, so `D` is not available at `g` — and
nothing needs it in advance.

> **T11 — Each level halves, over a span that widens by that level's reach.** An accepted bid holds more slots
> than it costs, and a price is at least 1, so it holds at least 2. **The assignment is a partition** (R23), so
> no two accepted bids hold the same slot — disjointness is definitional here. One accepted bid promotes
> exactly one unit, firing at the bid's own coordinate. Writing `A_k[a, b]` for the level-`k` activations in
> frames `[a, b]`, and `reach_t(k)` for D14's reach at that level:
> ```
> A_{k+1}[a, b]   ≤   ½ · A_k[a − reach_k, b]
> ```
> A bid holding only its own slot can never clear its price, which is what forces the halving. **This halving
> is what D14's schedule is derived from**, so the two are one statement read in opposite directions: the count
> falls because bids must cover more than they cost, and the reach grows because the count fell.
>
> **The span widens because coverage reaches back.** A bid firing at `b` covers activations as far back as
> `b − reach_t(k)`, so the ones that pay for a unit inside `[a, b]` need not lie inside `[a, b]` themselves. At
> `a = b` the right-hand side spans more than one frame, so **a single frame's count need not halve**.

> **T12 — How deep a frame can build.** Unroll T11 from `a = b = f`, widening by each level's own reach:
> ```
> A_D[f, f]   ≤   2^(−D) · A_0[ f − Σ_(k<D) reach_k , f ]
> ```
> Level `D` is active at `f` only if `A_D[f, f] ≥ 1`, so **a frame reaches depth `D` only when the base fired
> at least `2^D` times inside the span feeding it.** Since a frame holds at most one activation per
> `(dimension, position)` (D5) the base rate is bounded by the declared **slot** count `B` — dimensions times
> the extent they are laid out over. **Nothing is declared or capped: the bound is read off the alphabet and
> the reach, both already given.**
>
> **Whether it binds depends on `dim`.** With D14's schedule the span grows as `2^(D/dim)`, so for `dim ≥ 2` it
> grows slower than the `2^D` opposite it and the inequality resolves. For `dim = 1` the two sides grow
> together and this stops binding; T13 bounds depth there.
>
> **At `R_t = 1` the temporal span collapses and the spatial box carries it.** `2^D ≤ A_0[box]`, where
> the box is the region D14's reaches admit around `f`. MNIST is this case — `dim = 2`, one event dimension
> laid out over 28×28, so `B = 784` and `2^D ≤ 784` gives `D ≤ 9` once the box covers the frame. On real
> digits it binds tighter still, since about a fifth of the frame is ever active.
>
> **Deeper levels cost proportionally more base activity**, since each one adds its own reach to the span that
> has to supply the doubling. That is why a rich frame in a quiet stretch does not build deep: the exponent
> needs extent, not just breadth.

> **T13 — Depth is bounded by the run, logarithmically.** A neuron decides nothing until it has evidence, and
> `H` is how much (D24, R12). A level-`D` neuron fires at `2^(−D)` of the base rate (T11), so filling its ring
> takes `H · 2^D` of its channel's frames. After `F` frames the stack has therefore reached at most
> ```
> D   ≤   log₂( F / H )
> ```
> **This holds at every `dim`, and it is what bounds the temporal case where T12 does not.** It is the reason
> an unbounded file does not license an unbounded stack. Raising `H` makes the machine both more selective and
> shallower for a given run, which is one of the three effects D24 attributes to it.

**What contraction builds.** Each surviving bid contributes one unit above. The reduction is set by the data,
not the topology: a neighborhood the patterns describe well collapses hard; one full of surprise barely
collapses at all, which is the correct outcome for it.

**On R23 — why a proved minimum is not needed.** Choosing the accepted set that genuinely minimizes the frame's
sum is prize-collecting set cover. The file is never finished, and an election improves it the way a bill
improves a table — one step, taken against evidence that has already moved on.

**On D25 — an uncovered neuron does not make the file approximate.** It is stated, just not by a unit above.
Reading coverage as fidelity turns a pricing question into a correctness one.

**On R24 — the delay stacks and the memory does not.** Each level needs only its own coverage set,
`reach(k) + 1` wide in time and one box wide in every other activation dimension (D14, D26), and nothing global
is held for the prediction, which is an output and not a map.

# 12. The order of a frame

> **T14 — One pass resolves inside the frame.** A bid carries only backward neighbors (R20), so every election
> runs on frames already in hand, and the bill that fed it ran before it — bill, offer and election are all
> inside the level and inside the frame, which is why nothing in the loop costs latency in the stack. A unit
> promoted at `f` is available as an offset-0 neighbor to the level above at `f`, and its own connections
> forming later gate nothing. **Spanning patterns therefore cost no latency in the stack**; the only thing
> that settles late anywhere is R24's accounting, and nothing waits on it.

**Why there is no phase boundary.** Splitting the stack would declare a schedule of a different kind — one
reach below the boundary, another above it, and a transition wherever the lower half happened to stop firing
children. D14's reach also varies with level, but it is not that: it is one expression applied uniformly to
every activation dimension of every channel at every level, with the level entering only as the exponent T11
puts there. **The distinction is between a boundary and a formula.** A boundary has to be placed, and nothing
places this one; a formula is evaluated wherever it is read.

**Which makes the distinction emergent, which is the point.** Reach already emerges from the vote (R5) —
offsets where nothing recurs lose their slots. Under one stack a pattern *discovers* whether it is spatial,
temporal or mixed, rather than being whichever the phase that minted it allowed. A level-1 pattern naming one
neighbor in its own frame and one two frames back is an ordinary pattern, and there is no stage at which it
would have been unrepresentable.

**On R25 — there is no spatial stack that resolves before a temporal one.** A neighborhood names offsets, and
nothing in the rule distinguishes a spatial component from a temporal one, so the two are compressed together
by construction rather than in sequence.

**On R26 — why a flat top level would be none of those things.** The history writes exactly the frontier,
rewards credit exactly the frontier, the prediction is expanded from exactly the frontier and the next action
is chosen by exactly the frontier. In the worked drawing, `i` and `j` are covered by nothing, so they stand in
the frontier beside a level-3 pattern — whether that is because they offered no child or because the child
they offered lost its election makes no difference to the file.

**On R26 — why coverage silences.** A covered neuron is recoverable solely by expanding its coverer, and the coverer's
expansion already reaches everything the covered neuron names. So in the file it is a symbol already written; in the
prediction it is an expectation already placed, since the coverer's connections are the same future over a narrower
situation; and in selection it is the general case the coverer was minted to escape (R35). One rule, three readings,
and all three are the same rule against saying one thing twice.

# 13. The prediction

**Why the assertion went, and what the prediction is instead.** The assertion was the machine's forward claim
about the run: every uncovered unit's forward half, expanded, resolved to one owner per base slot by a vote
within a level and a cascade across levels, then scored against what arrived, with the misses written into the
file. It was the second most involved mechanism in the design after the adjustment, and it decided nothing —
no test read the corrections, no pattern was ever retired for them, and the election never saw a completed
span. What survived it is the part that is useful to whoever is downstream: the expansion. An apex unit's
connections say what it was followed by; expanding that through dictionary lines lands base symbols in frames
ahead; one winner per dimension over the apex is what the machine expects. It goes out. It is not a line in
the file, and it is never wrong in any sense the machine keeps.

**On R27 — why one winner per dimension and not a union.** A union is more information, and for a while the
design emitted one and left the reduction to the consumer. Two things argued against it. Every consumer the
machine is built for — a readout, an environment, a scorer — wants one symbol per dimension, so the reduction
would have been written on every side of the output instead of once inside it. And a vote across the apex is
already needed on the action side, where one action per dimension has to run, so the event side gets the same
vote for free with a different winning key. What the union was protecting against — a wrong pick being
scored — no longer exists, since nothing scores the output.

**On R27 — why the vote is by share and not by level.** The assertion used to give the higher level first
refusal on a slot and step down only when that level could not decide. That ranked descriptions: a unit that
chunked more of the frame spoke first about a slot inside it, whether or not it was the better predictor of
that slot. The vote reads no level. Every apex activation is one voter, normalized to one unit per dimension
so a voter hedging between two symbols is not two voters, and the symbol most voters expect wins. A coarse
unit that is usually right where a fine unit is usually wrong wins by being right, not by rank — and the
same is true in reverse. Nesting is already handled before the vote by D7: a covered activation does not vote,
so no unit ever votes beside the unit that subsumes it, and the double-counting the precedence rule was
guarding against cannot happen.

**On R27 — why the count travels down and nothing else.** An expected level-3 symbol expands to many base
symbols, each placed at the count the level-3 slot had. Re-weighting them — dividing the count among the
constituents, say — would make a unit that names more say less about each, which is backwards: the unit
expects the whole chunk, and each constituent is as expected as the chunk. What the normalization at the base
then does is per voter and per dimension, so a unit whose expansion lands two symbols in one dimension splits
one unit between them by their counts and no more.

> **T15 — Reach compounds.** A unit expected at `+2` may name something at `−1` of its own line, so expansion
> places a base symbol at `+1`, and one expected at `+2` from a unit whose pattern reaches `+2` places its line
> at `+4`. A reach bounds what a single pattern may **name**; it does not bound how far the machine can see.

# 14–16. Actions, reward, and selection

**On R29 — why the action cannot be at offset 0.** The events at `f` are recognized before the action is
chosen, so an action in their own column would be part of a backward half that is not yet in hand when they
are covered (D17), and a bid could name a neighbor the election has not picked yet (R20).

**On R31 — why the connection is a neighbor after all.** An earlier design held it apart: it crossed kinds, it
was temporal only, its ends need not sit at one level, and it was formed after every level had settled. Every
one of those is true of a forward neighbor in an action dimension. A forward neighbor crosses kinds whenever
an event neuron's connections hold an action; it is temporal because `Δt > 0` is temporal; its two ends are the
neuron and whatever fired, at whatever level fired it; and every activation connects after every level has run
(§10.3). What the connection had that a neighbor did not was an estimate, and an estimate is one more number
beside a count. So it is a forward neighbor with a strength and an estimate, and the neuron holds one set of
connections instead of two.

**On R31 — why the frontier is not enough.** Structure is recoverable by expansion, which is what lets the
file record the frontier alone; policy is not. Holding action connections at every level is also what makes the
ladder work: a level-1 pattern fires in many contexts and averages coarsely across all of them, a level-4
pattern fires rarely and averages sharply over one, and the estimate is waiting at whichever level ends up
uncovered.

**On R31 — why a covered neuron keeps learning.** Its estimates are the general case, the average over every
activation in its ring. Learning only while uncovered would make that average run over whatever no higher pattern
has differentiated yet, shifting with every new pattern.

**On R32 — why credit lands on the apex.** A committed higher action holds the dimension and suppresses its
constituents, so crediting the base would reward suppressed subordinates and calcify primitive-level policy.

**On R32 — why credit lands on the pattern and not its first step.** The estimate that selects a pattern has to
be what the pattern earned, which is the only reading that makes a multi-frame candidate comparable to a
single-frame one.

**On R33 — linear, not exponential.** An exponential fall reaches zero within a few frames, which leaves a
reward that arrives late attributable to nothing — and a reward that arrives late is the case an unscoped
reward exists for. A linear fall keeps a nonzero share at the far end of the span. The cost is that frames
which had nothing to do with the outcome take a share as well; those shares are the smallest ones, they
average out over exposures, and no structure is priced on them.

**On R33 — why the scope is the environment's to give and not the machine's to infer.** An environment that
can name the channel and the frame is reporting something it already knows, and there is nothing for the
machine to work out. One that cannot is not withholding information — it does not have it, and no amount of
machinery on this side would recover it. So the scope is an input with a default, and the default is the
honest statement of ignorance rather than a fallback path: the same arithmetic runs either way, over a wider
span and more channels.

**On R34 — why nothing weakens, and why nothing is windowed either.** Never taking an action proves nothing
about its worth, so an unchosen action must not be penalized, or the brain collapses onto whatever it tried
first. That is the first half. The second is that the estimate is a lifetime mean, and an earlier draft
rejected that on the argument that a lifetime is a second horizon beside `H`, which R10 forbids. It is not a
horizon: it is the absence of one, and it has no parameter. Under this rule `H` governs structure and nothing
else, which is one knob fewer. What a lifetime mean gives up is the rate at which one connection absorbs a changed
worth, which falls as `1 / strength`; what it gets is an estimate that is exactly the sample mean, and a distribution
that is specific because of who holds it rather than because of when it was written (R35). Attempts to buy
responsiveness inside the connection — decay, a window, a rate — were tried against the stock demos and lost to the
plain average every time.

**Three things this costs, stated plainly.** An action judged bad early stays judged: once every action in a
channel has a connection, nothing new is wired, and an action that was unlucky on its first samples is re-tried only
if it becomes the least bad (R37). That is the ordinary weakness of a greedy bandit, accepted for determinism.
A connection wired ahead of any exposure is one neutral pseudo-sample (R31): its strength is one higher than what
was seen, which is a prior of a single observation at zero, and the honest name for it. And a connection never
leaves: memory is bounded by distinct co-occurrences, which is the alphabet squared per offset at worst.

**On R34 — a neuron that fires rarely remembers exactly as long as one that fires often.** Both remember
everything. What differs is how many exposures each has, and so how far one new sample moves the mean. A
moment minted by replay (see [hippocampus.md](hippocampus.md)) holds its estimate across any stretch, at
whatever strength its few exposures gave it, with no aging law to argue with.

**On R34 — why reward cannot price structure.** A policy is not a description: the decoder replays the actions
the file records rather than choosing any, so nothing a reward says about an action changes what it costs to
state one.

**On R35 — recognition and execution run in opposite directions.** Events compose bottom-up; actions unfold
top-down, and selecting a high-level action pattern is a commitment to perform it. The two hierarchies connect
at every level, so an event neuron's connections can name an action pattern — a high-level situation joined to a
high-level response by a single connection, which is how a complex action sequence is learned as the answer to a
complex event sequence.

**On R35 — why the default is a connection on every neuron.** Every neuron is born holding the declared default at
every forward offset, base neurons included, so from the first frame every apex activation has an inference
and the walk (R37) can begin on the first negative reward. An earlier draft held connections on patterns only, which
left the base mute and exploration waiting for the first bought pattern. The rule that an unreached dimension
runs the default survives as a safety, and it fires only when nothing at all is on the apex, which is an empty
frame.

**On R36 — why there is no confidence correction.** The correction would be a parameter with nothing to
derive it from, and R37's walk is what buys the thin estimates their exposures.

**On R36 — why a covered neuron supplies nothing.** A pattern exists to tell one situation apart from the
general case its members fire in, and a member's estimate is that general case: an average over every
situation it has ever fired in, the pattern's among them. Letting the two compete puts the average the pattern
was created to escape back into the decision it was created for. The specific situation was recognized;
nothing general is allowed to speak into it. That a new pattern starts with only the default's estimate and
explores is right — the general answer is precisely the one just judged too coarse.

**On R36 — why a thin estimate displacing a worn habit is not a defect.** It is the exploration: a situation
only gets sampled by something being tried in it.

**On R36 — why level is not read.** An earlier draft resolved inferences by level first and estimate second,
on the argument that a higher action pattern decides more of the timeline. That let compression override
reward outright: a level-4 connection at a small negative estimate beat a level-1 connection at a large positive one,
which is a second place where the two objectives meet and the wrong one wins there (R34). The base-level vote
reads the estimate alone. A specific situation still tends to win, because a child's estimate is over one
situation and a base neuron's over many, so where the situation matters the child's number is the sharper one
— but it wins by being a better estimate, not by rank, and where the general case is the better predictor it
is allowed to be.

**On R36 — why the expansion carries the estimate unchanged.** A selected level-3 action expands to a program
of base actions, and each base action runs because the program was worth the estimate, so each carries it.
What the standing inference then does is hold the program's frames at that estimate against later frames'
fresh inferences, which is what "a plan holds because it keeps winning" means: the plan is a set of base
actions at one estimate, and any frame's fresh inference that beats it at one of those frames takes that
frame.

**On R37 — why always executing the best-known action is a problem.** An action that merely scores acceptably
can hold a situation forever. Thompson sampling over the action connections is the obvious probabilistic
alternative, and it drops into the same place.

**On R37 — what the walk buys, and what it does not.** It is deterministic, so a run reproduces and a
regression is a real regression. Other strategies drop into the same place, and swapping them changes no
structure. What it does not buy is a second look: the walk wires each action once, and with no window nothing
is ever forgotten and wired again, so an action's first few samples are the only ones it gets unless it is
selected on its own merit afterwards. In a stationary world that is the right economy; in one where an
action's worth changes, the neuron that notices is a new child with fresh connections, not this connection.

---

# 17. What is provable about compression

The claim the specification can make, and the one it cannot, stated once.

## Provable, per move, on the evidence in hand

Each structural move is a non-increase on the file it is measured against, evaluated over the population it is
measured on at the moment it is made. Strict where marked.

- **Add (R15).** Strict. The candidate joins iff its summed saving exceeds `1 + |C⁻|`, and R19 step 3 lets every
  activation do at least as well as the test counted, so the ring's file shrinks by at least the margin the test
  found.
- **Retire (R17).** Strict. A pattern's margin *is* the change in the ring's file on its removal: the history
  term rises by `Σ (covers − price)` and the dictionary term falls by `1 + |e|`, which is the margin with the
  sign reversed. Negative margin, shorter file.
- **Re-centering (R5).** Non-increase. Entering a slot changes the ring's file by `(n − 2 · count + 1)` over the
  pattern's population at that slot, dropping one by the negative of that, and R4 fires only when the sign is
  right and holds still at zero. This depends on D23 keeping `present` and `held`: with only the assigned share
  counted, the count that decides entry is not maintained and the claim does not hold.
- **Re-derivation (R6).** Non-increase, by construction: a re-derived cover replaces the held one only when
  strictly cheaper.
- **Cover (R18) and election (R23).** Each accepted pattern or bid names strictly more than it costs, so a
  covered activation or frame is strictly shorter than the same activation or frame stated flat, by at least one line
  per acceptance.

## Provable, per frame, against the best available

R18 and R23 are one procedure, and that procedure is ratio-greedy weighted set cover: each pattern or bid is a
set with cost `price`, and every slot also has a singleton set of cost 1, standing as its own line. Chvátal's
bound (1979) applies directly — the cover the greedy returns costs at most `H(n)` times the cheapest cover
buildable from the same sets, where `H(n) = 1 + 1/2 + … + 1/n ≈ ln n` and `n` is the largest set's size, here
the largest neighborhood offered. **So per frame the machine's history term is within `H(n)` of the best it
could have done with the bids it was offered**, and the one-pass election this replaced had no such bound.

## Not provable, and why

**Across frames.** Every frame adds an activation to each active neuron's ring, evicts one, and adds raw lines to
the machine's history before any election runs. Connections are outside all of this: they are in no file,
and no move is priced on them (R34). "The file is shorter after frame `f + 1` than after frame `f`"
is false of any compressor reading a stream, this one included. What holds is the statement above: never longer
than flat, and every structural move a descent on the evidence in hand.

**Across the two objectives.** D12 is explicit that the neuron prices over its ring and the machine over its
frames, that the numbers differ, and that they are meant to. A neuron's descent is therefore on its own
history, and a pattern that pays on the ring but is never bought is a dictionary line with no realized saving
at the machine's level. The design forbids the coupling that would close this — the election delivers nothing
to the neuron (R23, R25) — so **no single quantity is descended by the whole system**. Each side descends its
own file; nothing descends the sum, and a proof cannot be had without a rule the design deliberately does not
have.

**Beyond a local optimum.** Greedy cover plus coordinate descent — hold covers and take the majority, hold
neighborhoods and re-derive covers — reaches a state in which no single move shortens the file. It does not
reach the shortest file. Weighted set cover is NP-hard and `ln n` is within a constant of the best any
polynomial-time procedure achieves in general, so the `H(n)` ceiling is where every practical design stops, not
a weakness of this one.

## The theorem, as it can be stated

> **T15 — Every move the system makes is a non-increase on the file it is measured against, evaluated on the
> evidence in hand; the machine's encoding of each frame is strictly shorter than the raw frame whenever
> anything is promoted, and within `H(n)` of the best encoding available from what was offered.**
