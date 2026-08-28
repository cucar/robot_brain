# UCAR — theorems and commentary

Everything here elaborates [algorithm.md](algorithm.md) and none of it is normative. **T** is a theorem: a
claim that follows from the definitions and rules, stated with its argument. Everything else is commentary —
why the design is shaped the way it is, what the alternative would have cost, and worked examples. Items are
keyed by the D, R or section they belong to, in the order the specification introduces them.

---

# 1. The machine

**On D2 — why a child sits at offset zero.** A neighborhood is written relative to the parent's firing, so the
child sits at offset zero by construction. A centroid over what the child covers would leave the level above
differencing coordinates from origins its neighbors were never measured against.

**On D4 — a neighborhood is a box.** Adjacency is a conjunction, so something far away in one dimension is not
a neighbor however close it is in another, and no dimension can rescue what another has ruled out.

**On D4 — receptive fields grow with depth two ways at once.** Through composition, because a neighbor is itself
a chunk. And through the radius, because each level's units are sparser than the one below and have to reach
further to find each other at all. Neither is declared per level: the first is what a pattern *is*, and D15
derives the second from the first.

**On D5 — many firings per frame.** The bound has always been one firing per full coordinate. It reads
differently now only because the position used to be folded inside the dimension.

**On D6 — ages in a single neuron.** A neuron active at frames 10 and 12 is at ages 0 and 2 in frame 10 + 2,
then 1 and 3 in the next, and so on.

**On D7 — three consequences of having no rest value.**

- **Blank regions cost nothing** — no activation, no history, no dictionary, no compute.
- **A frame is a set** — variable-size, containing only what happened.
- **Absence still discriminates** — a pattern naming a hollow circle is measurably wrong on a filled disk,
  because the neurons inside the outline are present-but-unnamed and the distance counts them. Offness carries information
  without ever costing storage.

**On D7 — where sparsity comes from.** Whether a 0 pixel is a black event or nothing-to-see is the encoder's
choice, made where the input is produced. A dimension where something always happens is simply always active;
nothing depends on sparsity, it only profits from it.

**On D8 — what position-specificity would buy is fit, and that is what it costs.** One entry serving every
position describes statistics that differ by position, so it fits each worse than a position-tuned entry
would, and `d` is literally the corrections in the file (D17). The dictionary half of `L` falls and the
history half rises. **R12 adjudicates exactly that trade**, entry by entry, and no price anywhere changes for
it to do so: an activation costs 1 before and after, because the alphabet loses precisely the factor the
coordinate gains.

---

# 2. The objective

**On D9 — why nothing has to be honored.** An entry that goes is not a line the file must keep alive: the run
is simply re-encoded without that symbol, and the frames it used to cover are expressed by whatever the
dictionary now offers, with corrections where that is worse. An unbounded file costs nothing because nothing
was ever going to write it.

**On D12 — where compression is actually paid out.** A unit firing at `h` names `[h − reach_t, h + reach_t]`, so
writing it discharges up to `2·reach_t + 1` frames of the run at once. That is why a wider reach is worth paying a
wider line for, and it is why the normal is free and a child is not.

**On D13 — the two sums are what the two mechanisms work against, and that is the whole division of labor.**
Contraction works on the second over a given dictionary — the election is priced in exactly that sum, the
history half of `L` for the frames it can see, though it does not minimize it (§10.4). The one test decides the
first, entry by entry (R12). Neither can do
the other's job: the election cannot create or destroy a symbol, and a neuron cannot see what its symbol
saved.

**On D13 — why the neuron can compute what an entry is worth.** It knows what the file pays for an observation
it serves badly. With no child to propose, every neighbor stands in the file as its own symbol; with one, a
single unit discharges the whole neighborhood. That difference is the whole of the benefit, and it is a fit
against the neuron's own evidence.

**On D13 — conservative, and in a stated direction.** The line is paid once over the whole run, so an entry
that pays for itself within `H` of its neuron's own firings pays for itself many times over in the file. The
test asks for the stronger thing. What it therefore drops is structure that still describes the run but has
stopped describing the neuron's recent situation — which is the adaptation D25 is for, not an error in the
estimate.

**On D13 — a contribution is a bid's net, completed.** `cover − price` is what a bid states at the election
(R20, R21), over the backward half alone, because that is all a bid has seen. `credited − price` is the same
neighbors re-measured after the board spoke, over the *whole* span — the same backward terms plus the forward
mismatch the bid could not know. So the ledger's price is strictly the larger, and a bid the election took can
still settle negative. That is not a disagreement between the two: it is the half of the span the election
never saw, arriving when it can be measured.

So the benefit R12 sums is a ledger of per-observation nets — **not a tally of the election's verdicts**, which it
re-prices over a span the election never had. What the machine supplies is one fact per activation, the adjustment
(D28); everything else is the neuron measuring its own completed observation against its own neighborhood. The
neuron keeps the ledger and can invent no line of it, because the one term it cannot derive is the one the
machine hands it.

**On D13 — the test that used to sit here moved to where it can act.** The contribution once carried a gate:
zero unless the election would have taken the bid. It named a real condition — an unprofitable bid writes no
line, so it costs the file nothing — but it named it too late to be worth anything. The neuron can see
`cover > 1 + d_backward` at age 0, before it commits, so the same arithmetic decides *whether to route to the
child at all* (§9.1) instead of retroactively zeroing what routing already did. **So the contribution needs no
gate**: an entry is committed to only when that test came out positive on the half in hand, and what the bill
adds is the rest of the span and the board's answer, either of which can turn it negative. It stands as it
falls — negative when the entry covered the chunk and got its future wrong, and negative on the backward half
alone when it named less than it saw, which is what revives R13 at `R_t = 1`. Flooring the sign would make an
actively harmful entry indistinguishable from an unused one.

**On D13 — a negative settlement is transient or terminal, never an equilibrium.** Three mechanisms answer
one. Re-centering drops any forward neighbor wrong more often than right: R4's `1/2` is exactly where D10's
prices break even (T6), so every neighbor that keeps its majority is worth naming in expectation over the served
set, and what still settles negative is the minority tail of an entry positive in sum. The add test peels off
demand whose backward context is distinguishable, and both collapses come out cleaner for it. And an entry
that stays negative in sum falls below its line and is retired (R18). What none of them can reach is one
backward context genuinely leading to different futures: those observations share a bin (R7), so no mint can
ever separate them — the disambiguating information is not in the box at this level. There the collapse votes
the contested slots to silence, the entry stops naming them, and the events fall through as corrections on
the frontier (R31) — D26's covering-everything-is-not-the-goal, applied where the stream is incompressible at
this reach. A
level above, with wider reach, is what gets to hold the missing context. The one negative the vote cannot
shed is a neighbor another unit reliably covers — presence keeps its majority while the credit is gone — which
is the open seam in [algorithm-evaluation.md](algorithm-evaluation.md).

**Why a neuron's history states nothing.** It is evidence, not an encoding: it records what was seen so the
collapse can center on it (R4), and a decoder never reads it. The one place a price is needed — does this
entry earn its dictionary line — is a drop in `L` the neuron works out for itself, over its own evidence,
corrected by the one fact it could not have: which of its claims the rest of the level had already taken. So
exactly one thing crosses between scopes, and it is what keeps the arithmetic honest.

**Why predictive coding is load-bearing.** Under a literal history, a prediction that comes true saves nothing,
so no test could ever value prediction. Under predictive coding, being right is free and being wrong costs
corrections, so prediction error *is* file length, and it needs no term of its own anywhere. It enters in one
place only — D13's per-observation price, where the one test measures it against the neuron's own completed
observations (R12, R16). The election never prices it, because the election never sees a completed span
(R21).

---

# 3. Neighborhoods and distance

**On D15 — both ends are pinned by what is already given.** The declared radii fix the bottom. `log₂N₀` levels of
halving take `N₀` active neurons down to one, and at that depth the reach expression returns a reach spanning the
whole active region — so nothing has to state that the apex should see everything.

**On D15 — why every dimension grows by the same factor.** The thinning factor comes from a level's activation
count, and a level has one of those, so the schedule cannot tell the axes apart. Growing them all by
`(N_(k−1)/N_k)^(1/dim)` is the isotropic solution and the right default when nothing distinguishes them. Data
that chunks harder along one dimension than another thins faster there and would want that dimension's reach
to grow faster; measuring spacing per dimension instead of one count per level is what that would take, and
whether it is worth it is open ([algorithm-evaluation.md](algorithm-evaluation.md)).

**On D15 — why a schedule here does not cost the depth bound.** T9's span widens by the reach at each level,
so a reach that is a function of the level enters T15's unrolling. In `dim ≥ 2` the span grows as `2^(D/dim)`,
slower than the `2^D` on the other side, and the bound still resolves. In `dim = 1` the two sides grow
together and T15 stops binding; there T16 bounds depth instead, out of how long a level takes to fill its
ring. A channel with `R_t = 1` has no span to widen at all, and T15's spatial form carries it —
on MNIST that is `D ≤ 9`, far above the depth the data supports, so **activity is what limits depth and not
the theorem**.

**On D15 — the buffer walk is the whole of D6.** The activation is open for `reach_t` frames because that is how
long frame 10 takes to fall out of a `reach_t + 1`-deep buffer. A firing at 11 opens while this one is still open, so
several activations of one neuron are in flight at once, each at a different position in the buffer.

**On D16 — what made coarse offsets safe.** One neighbor per offset was a consequence of atomic offsets,
never a rule.
Logarithmic offsets make reach exponential in the alphabet, which is what makes the reaches D15 schedules
affordable to write down at all. Neither end of the offset alphabet is told which to use, and neither declares
anything (R4).

**A pattern is a name for a chunk of spacetime.** One pattern-learning algorithm and one kind of pattern.
There are four types only in the sense that two declarations cross: an **offset** may be zero or spanning in
any activation dimension, which is spatial against temporal, and a **neuron dimension** is event or action.
Neither is a separate mechanism. Spatial is a setting of the radii, and an offset does not know which kind it
is. Event and action are not separate for a plainer reason than it looks — the machine observes its own
actions, since each action dimension carries what was executed, so an action is a symbol read back the way a
pixel is and a pattern over it is learned by the same counting. **You could not tell from a dictionary line
which of the four you were holding.** The one asymmetry lives outside the pattern: events infer actions and
never the reverse, and that inference runs on reward (R38).

**On D17 — the distance is not a notion invented for matching.** It is literally the corrections that would
follow the activation in the file. A missed neighbor and a missed prediction cost the same, because in the
file they are the same thing.

**On D18 — the split is availability, not meaning.** A neighborhood names both directions symmetrically, and a
pattern is minted after its chunk has been seen. Routing is simply what cannot wait. This binds action neurons
no less than event ones: an action neuron fires when its action executes and waits out its forward half the
same way.

**At `R_t = 1` the vocabulary collapses.** `d_backward` **is** `d`, so the two distances of D18 are one;
`server_distance` **is** the minimum; nothing is ever in flight; the recording step has nothing to do; the bet
and the bill fall in the same frame; and bins key on the whole neighborhood. The add test still fires: with no
gate on the contribution (D13), a naming narrower than what it sees settles negative on the backward half
alone. Contraction loses its cross-frame
contention, since every claim spans one frame. Read the document with the temporal parts struck out and it is
the spatial algorithm, unchanged — the whole machine, not a stage of it, which is what makes a radius a
configuration rather than an architecture.

**Two different objects.** An entry's forward mismatch is what *this entry* got wrong against its own history —
the only prediction signal a neuron learns from. The machine's history counts what the *decoder* got wrong,
once per slot, over its single arbitrated assertion (§12). They coincide when one entry owns a slot outright
and diverge otherwise.

**On R2 — why the adjustment is stamped once.** The coverage set it was read from no longer exists, which is
R23's "no earlier promotion is re-scored" read from the neuron's side. Nothing here charges anything at the
moment it happened; the charge is still re-derived every time it is read.

---

# 4. State

**On D19 — why the two objects can never be one.** Being an L1 center (T1), a neighborhood is typically a set
no observation ever was.

**On D21 — why the record holds no absolute time.** Expiry was the frame number's only reader and expiry is
now a FIFO depth (R9), so the record has nothing absolute in it but the `position`, which the machine supplied
when it called — and which no comparison, price, count or vote ever reads.

**On D21 — an adjustment's `covered` half is a mask over the bin's key**, which every observation in the bin
shares, so the bin tallies it slot by slot exactly as it tallies the forward half. A settled observation having
no server of its own is what keeps a bin homogeneous (T3) however long its observations have sat there.

**On D21 — three lifetimes, and why they must not be confused.** Three notions of an entry stand in relation
to an observation, and they expire at different times.
```
committed entry   one activation, R frames    frozen at age 0    what it bid on and recognized from
server            until the distances move    the closest entry  what an observation costs
served bins       until the served set moves  inverse of server  what an entry aggregates over
```
An observation whose bin is handed to another entry is priced against the new one (R14), while the recognition
the machine still holds for that activation is the one returned at age 0. Holding the three in one field is what made them look
like one thing.

**On D21 — the adjustment is not a fourth lifetime**, because it stands in relation to no entry. It is a fact
about the board, recorded on the observation and read by whatever entry is being priced against it — including
entries that did not exist when it was recorded. That is what lets it price a newborn: coverage subsumes a
**neuron** (D27), so the adjustment would have been the same whichever entry the neuron had been holding.

**On D22 — why handover is arithmetic.** The distances the bins hold are the index R6 says nothing has to be
added to — an entry that moved recomputes its own distance to each of them, and every bin reaching for it is
current again. A bin moves whole (T3), so its tallies are subtracted from one entry and added to another in
`O(offsets)`, not `O(observations)`. The offset grid grows with the level, since D15's reach does, while the
number of neighbors in it stays fixed by construction — that is the invariant the reach is chosen to hold.

---

# 5. Counts, the collapse, re-centering

> **T1 — The collapse is the L1 center.** The result minimizes `Σ d(O, C)` over all sets, and the sum is over
> the whole span for every observation in it, because no partial observation is ever a neighbor. It is a
> *center*, not a medoid: synthesized, possibly a set the neuron has never seen. That is the point — it is the
> typical neighborhood, not a sample of one.

**On T1 — why a centroid will not do.** A centroid over sets is a fractional vector, which is not a set, cannot
be written into the file, and has no symmetric difference. The counts **are** the fractional object; the
collapse is how the design gets from it to something the decoder can expand.

**On R4 — the threshold is derived twice over**, from unrelated arguments: by L1 minimization for an entry's
neighborhood and a candidate's (T1) and by D10's prices for a slot's owner (T6), and it is the same `1/2` both
times. D10 prices the observation/claim difference exactly: asserting the wrong symbol costs 2, asserting
nothing and being surprised costs 1.

**On R5 — three consequences, and they are the point of the design.**

- **Neighborhoods track their demand.** An entry created on thin evidence is pulled toward its cluster.
- **Coincidence is voted out.** A neighbor present once loses its majority to silence and drops out.
- **Reach emerges.** Offsets where nothing recurs fall away. How far a pattern reaches is discovered, not
  declared. The radius bounds it; it does not set it.

---

# 6. The history

**On D25 — one count, so one denominator.** What R12 needs is not a shared clock but a shared divisor: an
entry's benefit is a sum over the observations it serves, and for two neurons' tests to mean the same thing
that sum must be over comparably much evidence. A uniform `H` gives that directly. A shared *window* gave it
only for neurons firing at similar rates, and gave a rare neuron almost nothing to decide on.

**On D25 — this is adaptation, not forgetting.** A neuron that keeps firing sheds its old observations as new
ones arrive, so entries describing a situation that has passed stop being served, drain their ledger and are
retired at the next bill (§8.2, R18). A neuron that falls silent sheds nothing: it holds its `H` observations
and its entries intact, indefinitely, and resumes from them when its situation returns. An active neuron
adapts exactly as fast as its evidence turns over, and a silent one simply waits.

**On D25 — `H` does three jobs.** It is the memory, it is R12's selectivity — double it and every entry's
benefit roughly doubles against an unchanged line, so more survive — and it is the rate at which the stack
deepens (T15). One number, three effects, all monotone in it, and it should be tuned knowing that.

**On R7 — both sides of the comparison are backward halves, but they are different populations.** A bin's is a
context that was actually observed; an entry's is a claim, the collapse of everything it serves (D19), and an
observation is routed to the nearest entry rather than to an equal one. Only when a bin's backward half happens
to coincide with an entry's are the two the same set. Were two entries allowed to share a backward half, the
tie would go to the older `id` every time and the younger could never serve.

> **T2 — Tallies are sufficient.** Everything the design asks of the history is a sum over slots, so the
> tallies answer it exactly. `d(b, C)` is `observations × d_backward` plus, at each forward slot, two for every
> observation holding a neuron other than the one `C` names and one for every observation holding nothing — or
> one for every observation holding anything, where `C` names nothing. Every observation in the bin appears in
> every slot's arithmetic, because none of them is partial. The collapse sums tallies. Nothing asks whether `c`
> at `+1` came with `d` at `+2`.
>
> **Adjustments tally the same way**, one counter per slot: how many of the bin's observations had that
> backward neighbor already covered, and how many had that forward slot taken. So a contribution (D13) is summed
> over a bin off the tallies exactly as a distance is.

> **T3 — The win test is per bin.** A candidate wins on `d_backward`, a property of the bin, so a bin is won
> whole. No bin is ever split.

> **T4 — The server is not the closest entry.** Routing chose on `d_backward`; the total is known `reach_t`
> frames later, and the entry that won the prefix can end up further in full distance than one that lost it.
> Nothing may assume the server was closest overall.

**On R8 — why records and not a summary.** An error total cannot answer retirement: when an entry goes, the new
server for its bins must be found afresh, which needs the bins and the distances they hold — a single number per
entry could not produce it. Keeping the ring is not a storage saving; it buys that the add test scans distinct
backward contexts and reads pre-summed tallies.

> **T10 — Every loop in the design is bounded, and none is capped.** A bill is a fixed number of scans and no
> iteration to a fixed point (R19). The pruning pass removes an entry from competition and creates none, so it
> runs at most once per entry; a retirement is collected within `reach_t` frames and takes its whole subtree in
> one step (R18). The election is two decisions and a settling, none of them repeated (R28). The level stack is
> bounded by the base activity behind a frame (T15) and by the run so far against `H` (T16), which together also bound the
> settlement walk (R26) and the depth of an assertion's expansion. Every bound falls out of a quantity the
> design already counts, so nothing has to be chosen to make the machine halt.

**On R11 — the trade, stated plainly.** Structure is not dropped for being old; it is dropped for having
stopped paying against the last `H` things its neuron saw. A neuron in a changed situation restructures at the
pace of its own new evidence, and one whose situation has simply gone quiet keeps what it had. Absence of
evidence is no longer read as evidence the structure died.

---

# 7. The one test

**On R12 — the adjustment is what makes a local test honest.** Without it the neuron counts neighbors another
unit has already covered and over-states every entry it holds. With it, the same sum is over credited neighbors
only, so a neuron whose territory a neighbor's unit took prices its entries at what they are actually worth:
near zero, because there is nothing left there to save. That is the whole of the correction the machine
supplies, and it supplies a fact rather than a number.

**On R12 — a contribution can be zero for two different reasons**, and both are the signal. Zero because the
neighbors were already covered — the file states this chunk some other way, and no sharper child here would
shorten it. Zero because the entry claims too little against its price — it would not clear the election. An
entry accumulating either drags itself toward retirement, and neither needs a mechanism aimed at it.

**On R12 — the movement of benefit is cheap.** The entry gaining or losing a bin and an observation joining or
being evicted are `O(1)` off the bin's tallies; a re-center is the walk the scan is already making (R19).

**On R12 — a newborn needs exactly the bracket and nothing more.** Where its territory was corrections, the
slots are free and it is promoted on its first recurrence — no line at the election means no deadlock at birth.
Where its territory is already covered, the adjustments say so, the credit never arrives, and the ledger
retires it. Both endings are decided by the machine's own records.

> **T5 — `L` is the objective, not a potential.** Nothing in the design descends it monotonically and nothing
> needs to. A handoff lowers the neuron's service cost, but service cost is a fit against remembered evidence
> rather than a term of `L`, so it carries no guarantee about the file; re-centering minimizes it over the
> served set while `|e|` may grow, so a single collapse can lengthen `L` outright. Nothing rests on descent
> either: a bill is a single improvement step and not an iteration to a fixed point (R19). Cross-frame churn is
> bounded by the negative-contribution trigger, the strict margin, and the fact that structure moves only at
age `reach_t`
> (R14) — and it is measured, not assumed.

---

# 8. The two moves

**This is facility location.** Observations are customers, entries are facilities, opening one costs `1 + |e|`,
serving costs the fit, and routing is the assignment. The opening cost is the only thing standing between the
design and memorizing every frame: if opening were free you would put a warehouse on every customer. The local
search is usually given four moves; here **split**, **merge** and **swap** need no machinery of their own.
Split is what add does to overloaded demand, merge is what delete does to redundant entries, and swap is add
followed by delete inside one bill — the child takes the bins, and the entry it stranded fails the pruning test
a moment later.

**On R14 — neither decision could sit at the other's age.** Recognition cannot wait, because the neuron has to
represent the frame it is in. Structure cannot come early, because a child names a whole `2·reach_t + 1` chunk
and that chunk does not exist yet — a test at age 0 would react to a recognition miss before knowing whether
the server was the right choice overall, and would mint against a pattern it has only half seen.

**On R15 — why the candidate is a center.** Incidental neighbors lose their slots and never enter `C`, so `|C|`
is the size of what recurs — which matters over a span `2·reach_t + 1` frames wide.

**On R16 — pricing `C` against a lagging server** would credit it a saving a third, closer entry already
delivers. The scan that corrects it costs nothing: the distances are already there (D22), and T8 puts the
re-derivation on exactly the passes that consume it.

**On R16 — why deciding the win set on full distance would be wrong.** It would count a candidate as winning
bins it can never be routed to — exactly those whose advantage is entirely forward — and overstate the children
most likely to disappoint. A bin can contribute zero because `C` claims too little against its price there,
which is the honest cost of a child routing will hand neighborhoods it serves badly: those bins pay for nothing
and the win set is smaller than it looks.

**On R16 — a candidate rejected today is not lost.** Every future negative contribution offers a new probe,
and re-centering
means a child that does get minted improves with exposure rather than freezing at whatever the probe happened
to be.

**On R17 — what makes release safe** is R18's condition rather than any wait: an entry is deleted only when
nothing is committed to it, so the neuron released has no open activations, and by the same argument neither
does anything beneath it. R14's "never pays off on the evidence that created it" is about the neuron; the entry
serving that activation's bin is not the same claim.

**On R18 — nothing scans for a dying entry outside the pruning pass**, and nothing has to: a margin moves only
on an event, and at a bill there are three.

1. **A child took its bins.** The add test's takeover strips it, and this is the pass that collects it.
2. **Eviction.** Served observations leave as demand drifts, and the entry is starved.
3. **The observation and its adjustment.** Both folded into the bin its server holds. The observation may be
   one the entry describes badly; the adjustment may say another unit had already taken the neighbors the entry
   was claiming credit for.

Re-centering moves margins too, but it is not a fourth event: it is what those three do to the counts (R5).

**On R18 — a candidate cannot be pruned by the pass that follows it.** Every retirement either hands `C` bins
or removes a competitor, so retirements only raise its benefit: its margin at the end of the pass is at least
what the add test priced. The order is therefore safe in one direction only, which is why it is fixed. An entry
that keeps winning recognition cannot postpone its own death by staying busy, because retirement stops it
winning anything.

**On R18 — what two sequential tests cannot reach.** A candidate that would pay *only* if some incumbent's
storage were refunded fails the add test and is never put to the pruning pass — two entries straddling one
cluster, each carrying its weight while the other stands, neither individually deletable. Pricing that case
would need a third move with a formula of its own, joint over an add and a retirement. It is not worth one.
Re-centering pulls an off-center entry to its cluster without being asked, and a straddling pair survives only
until drift or eviction starves one of them.

**On R18 — observations left behind need no repair**: a bin holds its distance to every entry, so the next
reader takes the closest of what remains (R6). An entry starved by eviction has no evidence left, and waits on
the neighborhood recurring.

**On R19 — two kinds of count movement, and each gets its own re-center.** Evidence moves counts in steps 1 and
2, and step 3 centers on them before anything is priced: measuring against a stale center prices a bin against a
neighborhood re-centering was about to move, and mints a child to remove a mismatch that was already going.
Structure moves counts in steps 4 and 5, and step 6 centers on those. **Neither re-center runs per span.** A
bill can fold several observations, so a center taken between two of them would depend on the order the inputs
gave them — an order the totals are indifferent to, and one nothing else in the design is allowed to read.

**On R19 — the trigger is a way to skip the scans rather than a gate on them**, and it is self-correcting in
the right direction: an entry serving many observations barely moves for one, so a contribution that is still
negative under it is a genuine miss; an entry serving three can swallow one whole, which is exactly the case
where a child should not be minted. **Only the add is skipped.** The pruning pass reads margins the fold and
the eviction have already moved, so it runs whatever the contribution was (R19).

> **T8 — The tests are the assignment.** The only consumer of a table-wide picture of servers is the add test's
> win set (R16), and the delete test reads it only through `served bins`. Both scan the whole table anyway.
> Everything else wants one bin's server: recognition takes the closest of that bin's distances (§9.1), the fold reads
> it for one bin, eviction reads it per departing observation. **So no pass exists to keep a global assignment
> current, and none is needed** — the scan that prices a move is the scan that makes it.

> **T11 — Nothing between bills can change an answer.** Counts move exactly when an observation completes, is
> evicted, or a served set changes (R3), and all three happen at a bill. Between bills every neighborhood is
> fixed, so every distance is fixed, so every closest and every price returns what it returned before. **A neuron
> between bills has nothing to recompute.**

**This is Lloyd's algorithm, interleaved with the data.** Assign points to the nearest center, move each center
to the minimizer over its assigned points: Lloyd 1957, better known as k-means. This is its L1 variant over
sets, with the collapse as the minimizer (T1), and `k` moves as add and delete change the entry count — which
is why those moves exist alongside it, since Lloyd only optimizes assignment for a given set of centers.

**What the design does not do is alternate to stability.** A bill absorbs its evidence, makes at most one
structural decision and re-centers on each: one improvement step, not a fixed point. Step 6 moves centers after
steps 4 and 5 fixed who serves what, so a bin can end a bill preferring an entry other than the one it holds.
Nothing repairs that inside the bill. Iterating would settle the table against counts the next bill moves
anyway, and every bill moves them. The table is never optimal over the ring and does not need to be. It needs
to be current for the next recognition, and that is one bin's distances.

---

# 9. The frame, per neuron

**On §9.1 — why the profitability test lives in routing.** A bid the neuron can already see is unprofitable
has nothing to gain from the election, and sending it up costs the level a claim to resolve for an outcome the
bidder could have predicted. Declining to bid is not a loss either: the normal carries the observation, and the
bill then asks whether a child should have existed for it (R13). Either a child takes the demand, or the normal
keeps it and re-centers on it (R5) — the same table adapting by a slower route.

**On §9.1 — why nothing comes back at the bet.** What the election settles about this activation is not settled yet
(T12), and nothing between now and the bill would use it. A neuron the entries describe only in general therefore
contributes nothing above it, which is R30's "fire no children" read one neuron at a time. At age 0 the neuron has
seen `reach_t + 1` frames of a `2·reach_t + 1` chunk, which is enough to recognize it and not enough to record it,
let alone judge it.

**On §9.2 — why the board is not read mid-span.** Coverage of the activation is still moving and stays so until
the bill (T12); reading a moving answer early would only be overwritten. A half-built observation is not
evidence, and the design never measures anything over a partial span (D18) — so there is nothing here that
could be measured, and nothing that would be true a frame later. Everything the activation has to say, it says
at once, when its span closes.

**On §9.2 — why the neuron is not called at all.** The band's only two jobs were transcribing arriving neighbors
and re-reading the committed entry, and neither is neuron work. Transcription is the machine's own frame data on
the way in, held for a reader forbidden to touch it until the bill — escrow, not custody. The re-read was worse
than idle: it let a standing recognition drift with counts moved by other activations' bills, which R24 already
denies ("a promoted unit **has spoken**") and which R14 already contradicted, since the activation was said to go
on naming *what it bid on*. Freezing the recognition settles that in R14's favour, and it is what §12 needs — a
claim that moved with the counts could not be rebuilt by a decoder holding only the active set. The call
disappears with the jobs: `process frame` reaches a neuron at ages 0 and `reach_t`, which at high levels is two
calls where there were `reach_t + 1`.

**On §9.3 — why the bill's decisions are once, not once per activation.** Deciding per activation would impose
an order on observations that are simultaneous — the pixel at one position did not happen before the pixel at
another — and the structure that came out would depend on it, which is the defect R28 removes one level up. The
order across the frame is forced rather than chosen at every step: an observation cannot be judged before it
has been recognized, and it cannot be priced before the board it was recognized against has settled.

**On §9.4 — T12 collapses to a single frame at `R_t = 1`**, settled and expiring at once, as everywhere else.
Every position billing in that frame folds first and the neuron decides once, exactly as §9.3 states it.

## One activation, across its frames

The three bands walked through on a single activation. Take `R_t = 3`, a neuron with the normal `N` and a child
`K`:

```
K names  {(a,−2), (b,−1), (c,+1), (d,+2)}
N names  {(b,−1)}
```

**Frame 10 — age 0, the bet.** `a` at −2 and `b` at −1 were present, so `K` matches the backward half exactly
at `d_backward = 0` while `N` misses `a` at 1. `K` wins, and the machine opens an activation committed to `K`,
holding an empty forward half. The neuron returns its **recognition** — `K`'s neighborhood, and a bid on `K`'s
child — so `c` is named at frame 11 and `d` at frame 12, on faith. **No bin was touched and no
count moved** — there is no observation yet, only a backward half and two frames to go. From here the
commitment is locked: the neuron has bid on `K` and may have been promoted on it. **The election resolves this
frame and returns nothing.** Whether `a` at −2 ends up covered, and whether this neuron itself does, can still
change for two more frames (T12).

**Frame 11 — age 1. `c` does not come; `e` fires in that dimension instead.** One thing happens: the machine
writes `(e, +1)` into this activation's forward half. **The neuron is not called.** The recognition it returned
at frame 10 still stands and still names `d` at frame 12 — it is not fetched again, and it would not change if
it were.

That is the entire frame. `K`'s counts do not move, `K` does not re-center, no distance is computed, and the
board is not consulted. `K` looks wrong here, but **"wrong" is not yet a quantity**: the chunk is half-seen,
and a half-seen chunk has no distance to anything (D18). `K` may yet be right at `+2` where `N` is wrong.

**Frame 12 — age `reach_t`, the bill.** `+2` lands and the observation is complete — `{(a,−2), (b,−1), (e,+1),
(d,+2)}`. Now, and only now, it becomes evidence: it finds the bin keyed on `{(a,−2), (b,−1)}`, whose server is
`K`, and the whole span folds into `K`'s counts in one step. `K` re-centers over all four offsets at once — if
most of its observations carry `e` at `+1`, the neighborhood flips `c → e`; if they are split, the slot loses
its majority to silence and `K` stops naming that offset. This is reach emerging, and it happens in one move
rather than being chased frame by frame.

**The adjustment is read here, on this frame and no other** (T12). Say the coverage set shows `a` at −2 taken
by a neighbor's accepted bid, and the assertion map shows the slot at `+2` held by a higher unit: the bin's
`covered` tally rises at `a`'s slot and its `superseded` tally rises at `+2`. From now on any entry priced
against this observation claims no credit for `a` and pays nothing for being wrong at `+2` — including entries
that do not exist yet.

Then the neuron decides. The contribution is negative — `K` named `c` and got `e` — so §9.3 runs in full. Had a
neighbor's unit covered this neuron outright, the adjustment would say so, `K` would claim no credit here at
all, and the candidate priced against this bin would fail on its own arithmetic — no rule about coverage
anywhere in the bill.

```
   frame 10  (age 0)         frame 11  (age 1)        frame 12  (age reach)
   ────────────────────────  ───────────────────────  ──────────────────────
   fire                      +1 arrives               +2 arrives
   route on d_backward       write (e,+1) into        observation complete
   commit to K, bid          THE NEURON IS NOT        the ADJUSTMENT comes with it —
   RETURN THE RECOGNITION:   CALLED. The recognition   settled this frame, gone the next
   K's neighborhood, naming  returned at 10 stands     enters bin, folds into K
   c at 11 and d at 12       and still names d
                                                      K re-centers, all offsets
                                                      add / delete / re-center
   ────────────────────────  ───────────────────────  ──────────────────────
   THE BET ─────────────── committed, collecting ───────────────── THE BILL ▶
```

**What does not happen is re-recognition.** The activation committed to `K`, and the recognition it returned
names `K`'s neighbors to the end, however badly `K` does. Two things may still move underneath it once its observation is in the history, and
neither is a re-recognition.

The **price** moves: as `K` re-centers on later bills, this observation and every other one in the bins `K`
serves are re-measured against the neighborhood `K` now holds (R2). An old observation that carried `c` and
cost `K` nothing may cost it two after the flip.

The **server** may move too: if a later bill mints an entry closer to this bin than `K`, or retires `K`
outright, the bin is handed over and the observation is priced against the new server instead. The activation,
if still open, keeps the recognition it returned on `K`, because it bid on `K` and that bid may be a live unit
one level up.

Neither is a revision of anything — it is the current table being scored against remembered evidence, which is
the only question the one test asks. The commitment is a fact about what was already done; the price is a
measurement taken now (R2, D22).

---

# 10. Contraction

**On the objective — the machine executes it, it does not evaluate it.** The file over one frame is the units
promoted plus what they got wrong. An earlier draft stated that as a rule of its own — accept a subset `S`,
minimize `cost(S)` — and named it prize-collecting set cover, which is the right classification and the wrong
picture: it reads as though something somewhere forms subsets and scores them, and the classification is only
interesting if you are choosing a search. Nothing searches. R28 resolves every slot in parallel on
profitability, drops the bids that do not clear their price, and hands their slots back to the accepted bids
that named them. **What the objective describes is the outcome of that procedure, not an instruction to
anyone**, which is why it belongs here and not in the spec.

**On R21 — `m` for a while stood for one side only, and that broke the add trigger.** The price counted the
neighbors an entry names and does not find, and nothing else. So a naming narrower than what it saw was never
charged for what it missed: an entry naming two of five settled positive and the demand for a child that would
have taken all five was never raised. With the error two-sided the same observation settles negative and the
add test sees it. The argument
for the one-sided form was that a neighbor present and unnamed costs a symbol only if no other bid names it, so
it is not attributable to this bidder. **The answer is that it is this bidder's box.** A unit predicts its
neighborhood; a neighbor that arrives unpredicted is written, and that is the error (D10, D17). What another
bid took is removed from the price by the assignment (R28 step 2), which is the honest form of the same
concern.

**On R21 — why the bid carries no estimate of the half it has not seen.** It could have carried the entry's
mismatch averaged over what it serves, and for a while it did. Three arguments against. R27 already promotes on
the backward match and lets corrections price the completed claim afterwards, so the estimate contradicts the
rule it sits beside. D18 forbids evaluating any quantity over a partial span, and a bid at age 0 has seen
exactly half of one. And the cost is charged anyway, measured rather than estimated, in D13's price at the bill
— so the estimate was a second charge for the same thing at a coarser scope.

**On R21 — so overlap eats surplus, not existence.** An entry naming ten neighbors with eight present bids
`cover 9` at `price 3` — it can concede five of its nine slots and still clear. Chronic overlap is priced at
the symbol instead: every conceded neighbor is credit the observation does not carry, the ledger drains, and R12
retires the entry. Occasional conflict costs a little; chronic conflict costs the line.

**On R21 — what dropping it costs, and why that is little.** The estimate acted as a fast brake: an entry
predicting badly had its individual bids priced up and lost them at once. Without it the brake is the ledger,
which is slower but exact. R16 will not mint a candidate whose measured forward errors on its own win set sink
it below `1 + |C|`, so a bad predictor mostly never exists; one that *turns* bad has each bad use subtract at
that use's bill, `reach_t` frames after firing, and R19 runs the pruning pass at that bill as it does at every
other one. What remains is an entry with banked surplus that starts
predicting badly and keeps winning bids until the surplus drains — which is the right answer, not a lag: while
its net over the window is still positive the file really is shorter for promoting it. A fast per-use brake
would be overriding a correct aggregate with a local one.

**On R21 — with the line in this price, promotion would be impossible outright.** A cover is at most the named
backward neighbors plus the bidder, so it never exceeds `|e| + 1`; a price carrying the line starts at `1 + |e|`.
`cover > price` could then never hold — not on a perfect match with nothing contested, let alone under overlap.
A test that asks one bid to pay an aggregate charge declines every bid.

**On R21 — there are not two clocks to reconcile.** R12 optimizes the dictionary against a neuron's own
history; this price optimizes one bid against the board's coverage. They answer different questions over
different evidence, and neither needs the other's span. What R12 does need is a denominator, and `H` is that
directly (D25).

**On R28 — why the board goes back down, and not the number.** A neuron can price its own entries — it knows
what the file pays for a neighborhood it names badly (D13) — and the only term it gets wrong is the one that
depends on what else was accepted. Contraction pricing the outcome and saying so is what keeps it a filter
rather than a second learner while still being the only authority on what was already covered.

**On D27 — these two windows are the whole of what a neuron ever learns from the machine.** D28 is a read off
them and nothing more, which is why it adds no state and no message anywhere: the machine already had to hold
both to run the election and resolve the assertion, and it hands the read down once per span. Neither window
introduces a parameter or a second buffer.

**On D28 — nothing is ever decided twice.** An election settles its own frame's assignment and is never
revisited (R23), so there is no in-flight coverage to maintain and no report to deliver between the bet and the
bill. The assignment simply fills in as later frames elect, and the machine hands the read over when the span
closes.

> **T12 — The bill is the one frame at which the adjustment can be read.** An activation firing at `g` names
> backward neighbors across `[g − reach_t, g]`. A neighbor at `f` can be covered by a bid firing anywhere in
> `[f, f + reach_t]`, so the last bid that can touch any of them fires at `g + reach_t` — and the neuron itself, at
> `g`, is coverable until exactly the same frame. **So backward coverage is settled at `g + reach_t`, which is
> the bill**, and the bill runs after that frame's election (R30). Every bid in the argument is at this
> neuron's own level, so one `reach_t` governs throughout.
>
> It is also the last frame at which the read is possible. The coverage set spans `2·reach_t + 1` and ages with
> the clock, so at `g + reach_t` it holds `[g − reach_t, g + reach_t]` — the activation's oldest neighbor sits on its
> oldest frame. One frame later it is gone.
>
> **Settled and expiring at the same instant, so the read has exactly one legal moment.** That is a third
> argument for the age the bill already sat at, alongside the observation being complete (D18) and the chunk
> existing to be named (R14).

**On T12 — the forward half is not settled there, and is read anyway.** A slot at `g + reach_t` can still be
claimed until `g + 2·reach_t`, which is `reach_t` frames past the bill — the same leading-edge shortfall §10.3
accepts for the election itself. Reading late is reading as much as exists, and reading per frame would buy
nothing: the assertion map is exclusive and re-resolved, so a later read supersedes an earlier one rather than
adding to it.

**On R25 — revision inside the window is free.** Encoder and decoder reach a slot with the same information and
apply the same rule, because neither reads one until nothing live can still claim it. A later claim that fits
better simply makes the file shorter. The slot is the unit of ownership, not the frame — units in different
dimensions never compete. One claim carries its slot, two that disagree carry neither, two that agree carry it,
three carry it when two of them agree. A split is not a decision to leave the slot empty — it is that level
having nothing to say. Nothing is compared across populations: two claims are counted, never weighed.

## The assertion map, mid-run

Each square is one base `(dimension, frame, position)` slot and holds the unit that currently owns it; the drawing
fixes a position and varies the other two. The coverage set is a different object and is not drawn here:

```
                                    frame
                     97    98    99   100   101   102   103
       dim A        ▓L1   ▓L1   ▓L2    L2    L2    L2     ·
       dim B        ▓L0   ▓L1   ▓L1    L0     ·    L1    L1
       dim C        ▓L0   ▓L0   ▓L2    ·     L0     ·     ·

       depth over     1     1     2     4     1     3     2
       settled?       ▓     ▓     ▓     ·     ▓     ·     ·

        ▓  settled — no live unit at any level can still reach this frame
        ·  still open — a claim may yet arrive, or one already here be outranked or outvoted
        L2 currently held by a level-2 unit
```

**Frame 101 is settled while frame 100 is not**, because the stack over 100 went four levels deep and the stack
over 101 went one. That is T13: the edge is ragged, it does not sweep, and there is no boundary to point at.
The drawing varies the frame because that is the axis it has room for; the same raggedness runs along every
activation dimension, and a quiet region settles while a busy one at the same frame does not.

**What settlement settles is the election, not the file.** It marks the point past which no further claim can
reach a frame. What the winning units then *expand to* is the current dictionary's business (D9): neighborhoods
keep re-centering underneath them, and a unit whose defining entry has been deleted stops being available at
all, so the run is re-encoded from what remains. **Contraction settles who covers what; D9 settles what that
costs to say.**

## What settles slowly

**How long after frame `g` before the file's statement about `g` can no longer change.** With `R_t = 3`, a base
neuron firing at 100 can be covered by a bid firing at 100, 101 or 102, so whether the history states it as an
apex unit or omits it as subsumed is not known until 102 has elected. If a level-1 unit took it, whether *that*
unit is itself apex is not known until its own `reach_t` after it fired, and so on upward. The file has a complete
encoding of the run at every moment, since D9 re-derives it — what is unsettled is not missing, it is still
liable to move. Coverage is elected once and never re-elected (R23); it is the assertion map that keeps
re-resolving (R25), and that is what the settlement condition is actually waiting on.

**On R26 — since the reach grows with the level (D15), the levels that settle last are also the ones that reach
furthest.** A bid accepted at `g + reach_t` can add a level after the fact, so `D` is not available at `g` — and
nothing needs it in advance.

> **T13 — What settles is a slot, and the settled ones are not a prefix.** Each slot settles once and stays
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

**Exact settlement of a single frame would need `4·reach_t + 1`.** Frame `g` can be claimed by units firing
anywhere in `[g − reach_t, g + reach_t]`, and those reach `[g − 2·reach_t, g + 2·reach_t]`. `2·reach_t + 1` therefore
scores bids at the leading edge against partially-visible competition. This is accepted — contraction mints
nothing that lasts, and it is the same shortfall D28 accepts when it reads `superseded` at the bill.

## The election

**Why per slot rather than by ratio, repeatedly.** A greedy pass over the ratio makes the same *kind* of
assignment — every covered neuron ends up credited to whichever bid the queue reached first — but it makes it
implicitly, by processing order, which is why it needs a tie-break clause to be deterministic at all and why it
cannot be run in parallel. Resolving each slot on its own merits makes the assignment explicit,
order-independent by construction, and parallel across slots. It also makes contraction the fourth use of one
primitive rather than a mechanism of its own (§5.2).

**On R28 — why the third step is not optional.** A promoted unit's neighborhood *is* its dictionary line
(R20), so expanding it recovers every neighbor it names, whether or not that neighbor was assigned to it. Coverage
is therefore a fact about what the accepted units expand to, and the assignment has no power over it: writing a
correction for a neuron some accepted unit already names would write a symbol the file already contains. Step 3
makes the bookkeeping match what expansion already did. Without it, a slot held by a rejected bid reads as
uncovered, its neuron credits itself for territory an accepted unit already holds, and R12's ledger runs hot.

**On R28 — nothing iterates, and the pass can only under-accept.** Steps 1 and 2 decide; step 3 only allocates
credit among decisions already made, so no acceptance can flip and there is nothing to converge to. A bid whose
territory is taken stops clearing its price on its own. The consequence is that the election can leave a neuron
to a correction a second look would have covered, and can never promote two units for one chunk — which is the
failure contraction exists to prevent.

**On R28 — a ratio that ranks is not a price.** `cover / price` is a selection score over two counts: it
estimates nothing, prices nothing, and no cost anywhere is set by it, so §14's claim is untouched. Both are
counts and a price is at least 1, so the score is a ratio of positive integers and the winner is simply the
largest.

**On R28 — why the coordinate tie-break is not decoration.** Two activations of one neuron bid with one
creation order, and in a solid region that is the common case, so without it the pass would have nothing left
to decide with.

**What is given up.** Greedy is the classical approximation for set cover, with slack against the optimum
bounded by `ln n`. Two fixed passes have no such guarantee, so the slack becomes unmeasured rather than
bounded. The case it loses is a boundary one — a bid that sheds most of its territory to better ratios, falls
under its price, and takes down a neighbor that would have cleared with the slot it took. Accepted
deliberately: **contraction mints nothing that lasts**, so a marginal cover costs a bounded handful of
corrections and nothing structural. Two chunks sharing a boundary neuron is how a stream tiles, and the bid
that loses that neuron simply counts one fewer.

> **T9 — Each level halves, over a span that widens by that level's reach.** An accepted bid holds more slots
> than it costs, and a price is at least 1, so it holds at least 2. **The assignment is a partition** (R28), so
> no two accepted bids hold the same slot — disjointness is definitional here. One accepted bid promotes
> exactly one unit, firing at the bid's own coordinate. Writing `A_k[a, b]` for the level-`k` activations in
> frames `[a, b]`, and `reach_t(k)` for D15's reach at that level:
> ```
> A_{k+1}[a, b]   ≤   ½ · A_k[a − reach_k, b]
> ```
> A bid holding only its own slot can never clear its price, which is what forces the halving. **This halving
> is what D15's schedule is derived from**, so the two are one statement read in opposite directions: the count
> falls because bids must cover more than they cost, and the reach grows because the count fell.
>
> **The span widens because coverage reaches back.** A bid firing at `b` covers activations as far back as
> `b − reach_t(k)`, so the ones that pay for a unit inside `[a, b]` need not lie inside `[a, b]` themselves. At
> `a = b` the right-hand side spans more than one frame, so **a single frame's count need not halve**.

> **T15 — How deep a frame can build.** Unroll T9 from `a = b = f`, widening by each level's own reach:
> ```
> A_D[f, f]   ≤   2^(−D) · A_0[ f − Σ_(k<D) reach_k , f ]
> ```
> Level `D` is active at `f` only if `A_D[f, f] ≥ 1`, so **a frame reaches depth `D` only when the base fired
> at least `2^D` times inside the span feeding it.** Since a frame holds at most one firing per
> `(dimension, position)` (D5) the base rate is bounded by the declared **slot** count `B` — dimensions times
> the extent they are laid out over. **Nothing is declared or capped: the bound is read off the alphabet and
> the radii, both already given.**
>
> **Whether it binds depends on `dim`.** With D15's schedule the span grows as `2^(D/dim)`, so for `dim ≥ 2` it
> grows slower than the `2^D` opposite it and the inequality resolves. For `dim = 1` the two sides grow
> together and this stops binding; T16 bounds depth there.
>
> **At `R_t = 1` the temporal span collapses and the spatial box carries it.** `2^D ≤ A_0[box]`, where
> the box is the region D15's reaches admit around `f`. MNIST is this case — `dim = 2`, one event dimension
> laid out over 28×28, so `B = 784` and `2^D ≤ 784` gives `D ≤ 9` once the box covers the frame. On real
> digits it binds tighter still, since about a fifth of the frame is ever active.
>
> **Deeper levels cost proportionally more base activity**, since each one adds its own reach to the span that
> has to supply the doubling. That is why a rich frame in a quiet stretch does not build deep: the exponent
> needs extent, not just breadth.

> **T16 — Depth is bounded by the run, logarithmically.** A neuron decides nothing until it has evidence, and
> `H` is how much (D25, R12). A level-`D` neuron fires at `2^(−D)` of the base rate (T9), so filling its ring
> takes `H · 2^D` of its channel's frames. After `F` frames the stack has therefore reached at most
> ```
> D   ≤   log₂( F / H )
> ```
> **This holds at every `dim`, and it is what bounds the temporal case where T15 does not.** It is the reason
> an unbounded file does not license an unbounded stack. Raising `H` makes the machine both more selective and
> shallower for a given run, which is one of the three effects D25 attributes to it.

> **T6 — Why a strict majority, again.** Resolving a slot in the assertion is R4's third population (§5.2), and
> the cut is the same. For an entry's neighborhood it is required by L1 minimization (T1); here it falls out of
> D10's prices instead, because a wrong symbol costs 2 and a missing one costs 1. Writing `q_p` for the leading
> claim's share of the population and `q_∅` for the share saying nothing:
> ```
> take p        q_∅·1 + (1 − q_p − q_∅)·2  =  2 − 2·q_p − q_∅
> take silence  (1 − q_∅)·1                =  1 − q_∅
> take p  iff  2 − 2q_p − q_∅ < 1 − q_∅   iff   q_p > 1/2
> ```
> **Silence cancels**, which is what makes the rule independent of whether anything abstains. The constant is
> derived twice, from unrelated arguments, and no new parameter enters.
>
> **Two claims that disagree therefore leave the slot to the level below**, and that is arithmetic rather than
> a default: at `q_p = 1/2` asserting costs 2 with even odds where abstaining costs 1, so silence at that level
> is strictly the shorter file.

**On R29 — what the alternative would cost.** Were the fold conditional on the election, counts would be
selected by elections that counts had produced, and the loop would close on itself with no fixed point — an
entry that wins would sharpen and win more, while a covered neuron's neighborhood froze at whatever it held
when it was last promoted, going stale by exactly the amount the world moved, and having nothing to bid when
its coverer is later retired. It would also break D25: the neuron's history would stop being a record of what
it saw, which is the identity that lets there be one file.

**What contraction builds.** Each surviving bid contributes one unit above. The reduction is set by the data,
not the topology: a neighborhood the entries describe well collapses hard; one full of surprise barely
collapses at all, which is the correct outcome for it.

---

# 11. The order of a frame

> **T14 — One pass still resolves inside the frame.** A bid carries only backward neighbors (R20), so every
> election runs on frames already in hand, and it settles before the level's bills read it — bet, election and
> bill are all inside the level and inside the frame, which is why nothing in the loop costs latency in the
> stack. A unit promoted at `f` is available as an offset-0 neighbor to the level above at `f`, and its own
> forward half completing later gates nothing. **Spanning patterns therefore cost no latency in the stack**;
> the only thing that settles late anywhere is R26's accounting, and nothing waits on it.

**Why there is no phase boundary.** Splitting the stack would declare a schedule of a different kind — one
radius below the boundary, another above it, and a transition wherever the lower half happened to stop firing
children. D15's reach also varies with level, but it is not that: it is one expression applied uniformly to
every activation dimension of every channel at every level, with the level entering only as the exponent T9
puts there. **The distinction is between a boundary and a formula.** A boundary has to be placed, and nothing
places this one; a formula is evaluated wherever it is read.

**Which makes the distinction emergent, which is the point.** Reach already emerges from the vote (R5) —
offsets where nothing recurs lose their slots. Under one stack a pattern *discovers* whether it is spatial,
temporal or mixed, rather than being whichever the phase that minted it allowed. A level-1 pattern naming one
neighbor in its own frame and one two frames back is an ordinary pattern, and there is no stage at which it
would have been unrepresentable.

**On R31 — why a flat top level would be none of those things.** The history writes exactly the frontier,
rewards credit exactly the frontier, and the assertion (§12) is voted by exactly the frontier (R31a). In
the worked drawing, `i` and `j` are covered by nothing, so they stand in the frontier beside a level-3 pattern
— whether that is because they fired no child or because the child they fired lost its election makes no
difference to the file.

---

# 12. The assertion

**How the assertion composes with contraction.** The two divide by scope. A bid is *priced* on its backward
half but a promoted unit *claims* its whole span, so contested forward slots within a level are settled by the
collapse (R25). The cascade resolves what contraction cannot see: a level-3 claim and a level-0 claim landing
on the same base slot once both are expanded. The backward half enters at exactly one point and no other: it
decides *who votes*. Coverage settles which unit represents an already-observed neuron, and R31a spends that
answer twice — the represented neuron is silent in the file and silent in the assertion. What it never does is
decide what is claimed about a frame nobody has seen; that is the collapse's, on the frontier's own terms.

**Why a vote within a level and a cascade across them.** Within a level the claimants are independent:
different neurons, different histories, and one does not contain another. Across levels the danger is that they
are not — the lower units may be the higher one's own **constituents**, and letting three level-0 claims outvote
the level-1 unit that subsumes them would count one body of evidence twice at two resolutions.

**R31a removes that case rather than out-voting it.** The electorate is the frontier, and no member of the
frontier subsumes another — a covered neuron is inhibited, so a constituent never stands beside its coverer in
the first place. What the cascade is left resolving is two *uncovered* units at different levels landing on one
base slot, which is a difference of granularity and not of nesting: the coarser description is heard first
because it decides more of the timeline (R38a), and a level that cannot decide hands the question down to the
finer one. So the double-counting argument is where the inhibition rule comes from, and precedence is what
remains once the inhibition has done its work.

**On R32 — the decoder can reproduce this.** No entry's statistics are read, no share is formed, and no two
populations are compared, so R27's stance holds: structure self-corrects through corrections, and the assertion
does not second-guess the election.

> **T7 — Reach compounds.** A unit claimed at `+2` may name something at `+1` of its own, so expansion places a
> base claim at `+3`, past the radius. A radius bounds what a single pattern may **name**; it does not bound how far
> the machine can see.

---

# 13. Rewards, selection, and actions

**On R35 — why the action cannot be at offset 0.** The events at `f` are recognized before the action is
chosen, so an action in their own column would be part of a backward half that is not yet in hand when they
route (D18), and a bid could name a neighbor the election has not picked yet (R20).

**On R36 — why credit lands on the apex.** A committed higher action holds the dimension and suppresses its
constituents, so crediting the base would reward suppressed subordinates and calcify primitive-level policy.

**On R38 — recognition and execution run in opposite directions.** Events compose bottom-up; actions unfold
top-down, and selecting a high-level action pattern is a commitment to perform it. The two hierarchies connect
at every level, so an action pattern's neighborhood can name event patterns — a high-level situation joined to
a high-level response by a single association, which is how a complex action sequence is learned as the answer
to a complex event sequence.

**On R39 — why always executing the best-known action is a problem.** An action that merely scores acceptably
can hold a situation forever. Thompson sampling over the connections is the obvious probabilistic alternative,
and it drops into the same slot.

**Distribution over time** is a separable policy ([global-rewards.md](global-rewards.md)). R35's chain credits
one frame back; the planned generalization spreads each reward across the preceding span with linear decay, so
distant antecedents keep nonzero credit under long-latency reward.
