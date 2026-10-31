# UCAR — the claim, the risks, and what is still open

What [algorithm.md](algorithm.md) commits to, what could go wrong with it, and what has not been decided. Each
risk states what would be done about it, so measurement has a decision attached.

---

# 1. The falsifiable claim

Nothing in the design optimizes for reward structurally, and nothing structural reads one. A pattern is charged
for the neurons it names beside the activation and never for what followed (D25); what followed, and what the
actions earned, strengthens its connections, enters no test, and is not in the file (D12). Reward is therefore read
off structure the compression built for its own reasons: a situation acquires an estimate of its own exactly
when a pattern is minted for it (R35), so the machine gets better at acting by compressing better — richer patterns
are bought, and each child's connections are the distribution of what its situation was actually followed by,
with what each run earned.

**There is no path at all by which a reward can change what structure exists.** If the reward earned still
tracks compression, it is the by-product thesis and nothing else.

So: **reward earned should track apex reduction across levels, with no part of the machine pursuing structure
for reward.** Instrument both and plot them against each other. If they move together the thesis holds
mechanically. If they do not, the coupling between compression and reward is where to look, and it is the
assumption everything else rests on.

The standing metric is **what the apex costs to write per frame (§3.9), paired with the dictionary size that bought
it**. It should fall with exposure on recurring data: early structure is provisional, re-centering consolidates it,
and R18 takes what is left.

---

# 2. Risks

**Neither the election nor the table finds the optimum.** The election (R24) is a heuristic for the frame's sum,
and the greedy pick builds one candidate at a time and keeps each that pays (D33). The table provably reaches a
local optimum on a fixed history (T7), and a local optimum is what it is: two patterns straddling one cluster, each
paying while the other stands, are never merged at their own level. **Diagnostic:** on a small neuron, compare the
standing file cost against an exact cover solved offline over the same activations and pattern count. The gap is
the basin.

**The neuron never hears what it sold.** This is the design's largest deliberate omission. The election writes
the coverage set and reports nothing back (R24), so a neuron prices every pattern on what it saw, whether or
not the machine ever buys it. A pattern that describes the neuron's activations well but is consistently outbid on
the same ground — a neighbor's child reliably takes the territory first — stays in the dictionary, keeps being
offered, and keeps paying its line in `L`. The design bets this is rare: a neuron finds itself in many
situations and wins some and loses others. **Diagnostic:** per pattern, the fraction of its bids bought over
its life. A pattern below some low fraction for a long stretch is the case the omission gets wrong. **Fallback
if it is common:** report one bit per bid, bought or not, back to the neuron on the response of its next call,
and let R18 read benefit over bought activations only. That is far cheaper than the per-neighbor
adjustment the design used to run, and it closes the case without touching what a pattern learns.

**A function that fits and always loses stays.** Worth is read from the neurons' histories (D57, R50), and a
neuron records that a relation recurred, not whether the election took the fit it stood in. So a function whose
relations keep recurring but whose fits are outbid wherever they are placed looks worth keeping: it stays in
the dictionary, is placed at every level its relations are returned at, and keeps the neuron it was given the
one time it was accepted. The design bets this is rare. A relation at a function's past end joins the function
instead of starting a longer competitor (R54); what follows a call is a function above it, not a replacement
(R50); and where a function loses because another covers its nodes, those nodes leave the board, its relation
stops being observed, its `with` slides out of the histories and it is pruned the ordinary way (R18). What is
left is two functions over the same bare ground where one always wins. Whether that stays rare, or compounds
into neurons that fit and never cover, the runs will have to show. **Diagnostic:** per function, fits placed
against fits accepted over a stretch of frames, and the interneurons allocated that have not fired in a long
stretch. **Fallbacks if it is common:** the one bit per fit of the paragraph above, accepted or not, returned
to the neurons that answered for it, so that `without` counts a declined fit; or a function whose interneuron
has not fired within some stretch of frames leaves the dictionary, which is a timer the design otherwise does
not have.

**Earlier bidders win shared past neurons.** A bid at `f` keeps a neuron at `f − 2` against a better bid at
`f + 1` (R23). The bias is on the neurons at the boundary between what two children cover and it is stated as a cost.
**Diagnostic:** how
often an accepted bid's tally lost exactly one activation to an earlier frame's credit, and how often that activation
would have flipped the acceptance. If the second number is not small, the alternative is re-electing the past
within the coverage set's window, which is bounded and has been rejected so far for the ripple it sends up the
stack.

**The offer is the cover, and the board is not the history.** A neuron bids the patterns its cover took over
its own residual (R21); the machine covers the board, where earlier frames' children already hold activations, so a
pattern the cover passed over can be the board's best purchase and is never shown to it. This is accepted as the
design: the neuron optimizes its history and the machine its window. **Diagnostic:** per level, the activations the
election left uncovered that some pattern in a bidder's table names. That share is the cost of the split, and
nothing is done about it; it is measured so the cost is known.

**A cover is never re-derived, so it can go stale.** Recognition only covers the residual (§6.2), which is what
makes the call a descent (T7). It also means an activation saved under an old table keeps the patterns it was
given for as long as they own something there, and two activations with one neighborhood can be covered two
ways. The counts a pattern re-centers on are then partly the table's past. **Diagnostic:** the share of
activations whose cover differs from the cover D28 would derive fresh over the whole neighborhood, and the cost
difference. If the share is large and the difference is small, staleness is harmless; if the difference is
large, a periodic fresh derivation is the fix to weigh.

**The base votes its marginal.** A base neuron on the apex infers from its own connections, which is the
average over every situation it has ever fired in (D25, R35). Early in a run that is every voter there is, so
the machine's first actions are chosen on estimates that average over situations the base neuron cannot tell
apart. Coverage silences a base neuron
only once something more specific is bought over it. **Diagnostic:** the share of apex voters that are base
neurons, per frame, over exposure; it should fall as the dictionary fills. If it stays high on data that
recurs, patterns are not being bought where the base is voting, and the compression side is where to look.

**An action judged bad early is not re-tried.** A connection never leaves (R31), so the walk wires each action once
and never again (R37); an action that was unlucky on its first samples keeps that estimate and runs again only
if it becomes the least bad of a dimension where everything has been tried and the action policy outputs
the least bad (D51). In a stationary world nothing is
lost. In a world where an action's worth changes, the connection cannot notice — the design's answer is that a new
child with fresh connections notices instead (R34), which holds only where structure is actually being minted
over that situation. **Diagnostic:** per channel, the share of connections to actions whose estimate is negative on
fewer than three exposures and which are never selected again; and, on data where an action's worth is known
to change, how many frames pass before the apex voter for that situation is a neuron minted after the change.

**An estimate moves at `1 / strength`.** With no window, a connection with a thousand exposures moves by a
thousandth of each new sample, so a base neuron's estimate on stationary-then-changed data lags for as long
as it has history. The bet is the same as above: the general case is allowed to lag because the specific
case is a younger neuron. **Diagnostic:** for a change introduced at a known frame, the estimate at the apex
against the true worth, per frame after the change, split by the age of the voting neuron. If old voters
still decide the dimension long after the change, the hierarchy is not minting where it is needed, and the
compression side is where to look.

**The frontier need not thin.** A cover of `m` functions promotes up to `m` children at one coordinate (D8, D28),
and every variable a bid carries adds a value neuron beside them (D45), so the frontier above an activation can
hold more neurons than the neighbors the bid covered. Nothing caps `m`; what bounds it is that each extra child is
another line and another call, and the election stops buying the moment one does not pay. The compression is in
what the frontier costs to write, not in how many activations stand on it (T11). The reach does not rest on
thinning (D4), so what this costs is the size of the neighborhoods above: a neighborhood is the frontier within
reach, and it is as full as that frontier is fine. **Diagnostic:** bought bids per activation and frontier
neurons per frame, against neighbors per activation at each span.

**Far placement is lossy, and the loss compounds with height.** An offset is kept to one significant binary
digit (D6), so a neighbor 11 frames back is written as 8 and expanded to 8 (R28). Each level down adds its own
group's slack, so a high neuron places the base symbols of its farthest neighbors only to within the sum of the
groups along the path — its connections can say what came next but not, above a few levels, exactly
when. The file says so (§1); what it does not say is how deep the stack stays useful before its far placements
are too coarse for a program (R30) to act on, which is a cap on useful depth that nothing declares.
**Diagnostic:** per level, the placement error of expanded base actions against where they ran, split by
offset group; and the level above which a connection's base placements no longer win a dimension. If that level is low, the coarse groups are earning their reach on the backward side and losing it
on the forward, and a finer offset alphabet above some level is the fallback.

**Siblings agree until one is bought alone.** Two children bought at one coordinate see the same actions
follow them, so their connections agree until the first activation where one is bought and the other is not (D25). Two
patterns always bought together never diverge at their own level; the level above is expected to merge them
(§7 of the remarks). **Diagnostic:** for pairs of children of one neuron, the overlap of the frames they were
bought in against the overlap of their connections. Pairs high on both for a long stretch are the merge the level
above has not made.

**A newly minted child infers little for a while.** Its connections start forming at its first purchase and
only in the frames it is bought in (R17, D25), so until it has been bought a few times it infers from one or
two exposures. Until it has any, what it covers speaks in its place (R48), so the situation is answered by its
parents' habits rather than by nothing; whether those habits are a good enough first guess for the more
specific situation the child names is the question. **Diagnostic:** frames from a child's mint to the first
time its own inference wins a dimension, against how often it is bought, and how the reward under its parents'
choices compared with the reward once it chose itself.

**Nothing retires a pattern for a useless future.** A pattern is priced on what it names that did not fire
beside it, and never on what its child was followed by (D25, R18). So a child whose connections are worthless keeps
its line as long as its parent's pattern covers, and the machine's inference from it is noise. Nothing clears
a bad connection: the connections keep every action that ever followed, and the claim is that the estimate
sorts what the count cannot clear — an action that never paid carries its mean, the vote takes the largest
estimate, so a connection decides a dimension only where it is the best-paying thing any voter holds — and that
a pattern whose *backward* half is good is worth its line regardless. A voter with many connections still
places something in every dimension it has ever seen, so it never falls silent. **Diagnostic:** per level, the
share of frames in which the winning estimate rested on fewer exposures than the runner-up's, and whether the
reward that followed favored the winner. If the thin estimate keeps winning and paying less, the vote is passing
noise the machine could have withheld.

**The readout is unvalidated, and the position now lives outside the symbol.** Compressing harder can produce a
worse classifier, because a readout may be living on exactly the position-and-class-specific duplicates that
compression deletes — and D11 deletes them by construction rather than incidentally. The information is not
lost: it moved into the activation coordinates the body states. But a readout reading bare symbol identity
sees a translation-invariant bag and loses every bit of *where*, so it has to consume `(symbol, coordinate)`
pairs. A regression here will look like the compression was wrong when it was the decode. The readout gate in
[algorithm-implementation.md](algorithm-implementation.md) is the check.

**History size and reach sensitivity.** Every decision is exact with respect to the last `H` activations and blind
beyond them. `H` too small and patterns form on coincidences and the stack deepens faster than the evidence
warrants; too large and a neuron follows a moving situation slowly and keeps more structure than earns its
keep. Reach too small and no pattern spans what recurs; too large and every neighborhood is mostly noise at
build time. Measure both early and jointly — they interact through `|p|`, not through D27, whose denominator is
the same at every offset (D27). **Diagnostic:** how often the outermost offset is named against offset 0, swept
over `H`. If the outer reaches stay empty at every `H`, the reach is bigger than the data supports and
evidence is not what is limiting it. Sweep depth against `H` in the same runs — T13 makes the two move
together, and conflating them is easy.

**Boundary flicker on stationary input.** The ring is a FIFO and its population moves at every bill. A neighbor whose count sits
at the boundary follows the activations entering and leaving, naming it raises the pattern's price wherever it
is absent, that can drop the pattern out of a cover, and the smaller population re-decides every other neighbor.
The claim is that on stationary input this is flicker around a fixed point, confined to boundary neighbors, with an
amplitude that does not grow with run length — a claim about noise, which nothing in the rules proves. **This is
the standing test.** Per pattern per bill: neighbor flips against the neighbor's distance from the boundary,
`count · worth − (s − count) · log₂ |p| − reference` (D27), and cover changes per bill for the cascade. Expected: flips concentrated within a step or
two of the boundary, at a rate that settles once the ring is full and does not drift. A flip rate that rises with
run length, or flips far from the boundary, is the churn engine and a bug. Early tests are also decided by very
little evidence, so read the same numbers over the first thousand frames and again in steady state.

**Small builds.** A relation seeds once it has been counted twice (D33), so a candidate can be built on a
population of two or three, and the collapse over a small population names most of what it holds. Re-centering
largely defuses it — the pattern is pulled toward whatever recurs, or starves. **Diagnostic:** patterns retired
within a few calls of being built, against the size of the population they were built on.

**Shared patterns fit every position worse than tuned ones would.** D11 pools activations from everywhere into one
pattern, so a pattern describes statistics that genuinely differ by position and fits each of them worse. That
is a real cost and it is paid in charges, which is the body half of `L` — the dictionary half falls in
exchange, and D30 is what weighs the two. The design commits to the trade being worth it and offers no way to
buy back position-specificity except declaring a coarse position as a *neuron* dimension. **Diagnostic:**
charges per activation against dictionary size, before and after, on the same data.

**The cover pass is a greedy set cover, not a nearest-neighbor lookup.** One scan of the table per round, and
a round is one pattern taken (D28). Cost is `O(|cover| · |table| · |O|)`, and `|cover|` is exactly the
quantity the multi-child risk above says is unbounded. **Diagnostic:** cover-pass scans per activation against
cover size, per level.

**Routing cost.** `|O|` is set by the reach and by how fine the frontier is within it (D4, D5), and the cover
pass prices every pattern against it every frame. A child of long span standing among uncovered base neurons sees every
one of them within its span plus `R`. **Diagnostic:** scan volume against `|O|`, by span.

**An early partition can freeze.** A pattern never acquires a neighbor another pattern of the same cover already
holds — the neighbor's population excludes those activations entirely (D27) — so patterns grow into the residual and
never into each other. That is what stops two patterns converging, and T7 rests on it, and it also means a bad
early split of one stretch of the run across two patterns is not repaired by re-centering. It can only be repaired by one
of them retiring and the other growing into what it left, one per bill. **Diagnostic:** how often a retirement
is followed within `H` activations by a surviving pattern growing into the vacated neighbors. If it is rare, the
partition is sticky and R14 is carrying more of the load than intended.

**Election slack, bounded but unmeasured.** R24 is ratio-greedy weighted set cover, so its slack against the
best cover buildable from the same bids is bounded by `H(n)` and no better
([algorithm-remarks.md](algorithm-remarks.md) §9). The bound is worst-case and
says nothing about the slack on real frames, and since apex-neurons-per-frame is the headline metric, slack and
real structure are conflated in it. **Diagnostic:** solve one small window exactly (ILP) and compare, which
locates the realized slack inside the `H(n)` ceiling.

**The composition gap.** Both scopes price in one currency against the file (§3.9), and the neuron's sum and the
machine's are taken over different sets and are meant to differ (D22). What remains is that candidates are
*generated* locally: a demand no neuron proposes is a symbol the election never gets to consider, and no neuron
proposes one whose value lies in what it would let a *different* neuron stop paying for. Distinct from election
slack, which measures the election against a perfect election over the same bids; this measures propose-then-elect
against optimizing dictionary and frames together. **Diagnostic:** over a short run on one small level, compare the
file this design writes against the file a joint optimization over the same activations produces. That gap decides
whether contraction should stay purely a buyer or start supplying candidates back into the tables it covered. The
the neurons under one stretch of the run each build their own line for it; where those lines tie on the board they
now share one
child (R43), and where they are only near each other they still do not, which is where a constructive variant would
pay first.

---

**When a frame's numbers are final.** Settlement is a property of one activation at one full coordinate. **Nothing here delays anything the machine does**
— no pass blocks on it and no decision is deferred by it. **The only consumer is measurement**: when `L` or
apex-neurons-per-frame is read, the settled frames are the ones whose numbers are final.

**Frontier membership settles within the longest reach.** Whether an activation at frame `h` is covered is
decided by bids firing no later than `h` plus the longest reach in time among the neurons the machine holds,
since no bid reaches further back (D24).

**A frame's encoding settles at the top of whatever stack reached it.** A neuron one level up, firing later,
can name a lower neuron that names frame `g`. **Frame `g` is settled when no level holds an open activation that could
still join or leave that set** — a closure over the levels, evaluated upward.

**`D` is reached, not known.** The walk stops where a level accepts no bids and therefore produces none above
it, so the reaches summed up the stack bound a condition rather than counting out a delay.

---

# 3. Open questions

**Patterns and variables (D37–D47) — what stands between the model and an implementation.** A pattern above the
base names events and actions together (D5), and there is one kind of connection (D25). A pattern is of one of
two kinds, a function or a variable (D38). A variable is one set, of one of two types that mean opposite
things: a class is a set of members, one of which stands where a function names it, and a parameter is a set of
positions, offsets from its first, at all of which one and the same neuron stands. A function is the things it
names, each at an offset from the others: neighbors, variables of its table, and the owner's activation where it
names it; classes are its local variables and parameters its arguments. A function is a set of relations
between its nodes (D15, D34), so one that names the owner holds at the owner and one that does not holds
wherever its nodes stand in the window. Every level is explained as a set of function calls with their
arguments: an accepted bid is a call, a function's child placed over what the bid covered, at the latest frame
and the center of the box (D53), with one value neuron per variable where each value stood, or a lone
parameter's value neuron (§6.4, D45). A bid covers what its patterns name, the bidder
among them where a pattern names the owner (D31), and every symbol is counted at what it cost to write (D13).
Base actions take
no arguments, and one whose channel has no layout acts at a focus the environment holds; the variables are built first, each by its own collapse
and priced on its own uses, and the functions over the rows as the variables rewrite them (D27, D33); an
inferred pattern runs its actions and expects its events, weakly (R30). Nine cases are worked by hand on it
([algorithm.md](algorithm.md), Part VI). What is still open, in the order it bites:

- **The worked cases have not been re-priced.** Every symbol is now counted at what it cost to write, a base
  neuron among every neuron the machine holds and anything above it at what its bid wrote (D13). The cases were traced for
  whether each pattern still pays, and each does, but only the alternating pair carries numbers. What the new
  prices do to how soon a pattern forms is not measured: a table line names neurons among all the machine
  holds, and an occurrence now saves only what its neighbors cost, so a pattern needs more repetitions within
  the history than it did. **Diagnostic:** occurrences a pattern needed before it was added, against the size
  of the history.
- **Values under one function, above.** A function over two or more variables fires its child with a value
  neuron per variable beside it (D45). A level that only renamed them would cover what it writes and save
  nothing, so it is not built (D13). What can pay is a function in a value neuron's own table that names what
  always stands beside it. Whether, and how fast, the separate value neurons are put back together into one
  neuron that way has not been traced under these prices.
- **A lesson that depends on values together.** Where several variables hold under one function, each value
  neuron learns what followed it, and the vote adds their tendencies. That is right where the answer follows a
  majority of them, and it says nothing where the answer depends only on the combination, as the answer digit
  of a sum does. There the value neurons each infer the patterns that have covered them; the match proposes them all,
  since a neighbor found credits more than a neighbor missed charges (R49), and the pattern every value neuron
  names takes every share where the others take one, so it wins by share (R47). That has been
  reasoned on the addition case taught mixed and not run. Taught one case at a time
  ([algorithm-addition.md](algorithm-addition.md)), the machine has a neuron per combination from the first
  level.
- **A child fired frames back.** A pattern that does not name its owner fires its child where its things stood,
  which may be frames before the one being compressed (D53), and its first connect call hands it the frames it
  missed from the kept apexes (§7.1). How often that happens, how far back, and what it costs to catch up have
  not been measured. **Diagnostic:** the distance in frames between a child's coordinate and the frame it fired
  in, per frame.
- **Uses are places.** A relation and a pattern count uses, sets of coordinates, and the same coordinates seen
  from several activations of the owner count once (D27). The tallies the implementation keeps are per
  activation today, and telling one place seen twice from two places that look alike needs the coordinates of
  the seed's things on every count. Whether that stays cheap at a radius of a hundred has not been tried.
  **Diagnostic:** rows per use, per pattern.
- **Depth on input that repeats.** A stretch that repeats exactly fires the neuron minted a lap ago rather
  than a new one for it (R43, D36), and no chain of children closes on itself, since no line names a neuron
  built over its own child (D27). Whether the stack then stops growing, and at what depth, has not been worked
  out or measured: the six-bar cycle that settled at six neurons did so through closed chains, which the design
  no longer allows. Also open is how the first lap's neighborhoods, whose windows reach back past the start of
  the run, shape the children that form first. **Diagnostic:** the highest level reached per frame against run length; it should rise and
  flatten on input that varies and on input that repeats alike.
- **Prices move.** Every price is `log₂` of an alphabet (D13), and the alphabets move: `n` as the machine fills and
  is pruned, a table as entries come and go, a variable with its members. A neighbor is counted at what it cost
  when it stood, which is kept with it, so the same neuron can stand in two rows at two costs. So a pattern can
  cross zero with no change in its rows. **Diagnostic:** retirements whose cause was a change in an alphabet alone.
- **Every relation is tried in every call.** Keeping the relations counted (D47) costs `c²` per row entering or
  leaving for the relations within a row, `c` its neighbors, and `c · d` for the offsets that vary, `d ≤ H` the
  distinct neurons seen at an offset. Each relation tried costs a collapse over its uses, `O(H · c)`, and the
  greedy pick tries every one in every call (D33). How to try fewer without passing over one that would pay is
  open.
- **Near-duplicates stay apart.** Children are shared only between lines that stand for exactly the same thing
  (D36, R43). A pattern one neighbor off gets a child of its own, and two children that stand
  for nearly the same thing are never merged. **Diagnostic:** pairs of children whose accepted bids cover
  mostly the same activations, per level.
- **Membership lags.** A class fits only a member (D41); a non-member where a function names it is a failed neighbor, and it
  joins when it has recurred enough to pay for its entry (D27). Until then a new variant is written with failed
  neighbors. A parameter has no such lag: whatever stands at all its positions is its value. **Diagnostic:**
  occurrences of a function written with a class's failed neighbors, per class.
- **The child's place moves.** A child is placed at the latest frame its bid covered and the center of the box
  in space (D53). Re-centering that adds or drops a node at the edge of the box moves where the next activation
  of the child fires (D29), and what stood beside it a frame later is then at a different offset in the
  connections. Nothing shows how often that happens. **Diagnostic:** changes in a child's coordinate relative to
  its bidder, per pattern per call on stationary input; they should die out.
- **Reach bounds what a parameter can carry.** A parameter binds from offsets within its owner's reach (D4). A
  value needed from further back than that is out of sight, and a value carried forward is carried only as far
  as its owner's reach.
- **A function is callable only from a situation whose window holds it.** A voter can start a program only
  from an offset at least as far out as the program is long (R36), and an activation is open for `reach_t`
  frames (D9). Below that height the same behavior is dispatched a step at a time, each step inferred by the
  situation the last one created, with the recent past in the pattern carrying what the step needs to know.
  That works only when every decision's facts are within reach of the situation that makes it.
- **A situation that recurs unchanged runs again.** If a step leaves what its situation sees exactly as it was,
  the same situation fires and the same step runs, forever. Only a negative reward and the walk break it.
- **Loops have no member.** A constant repeat is unrolled, and nothing can say "down, three times". A run-length
  neighbor would be decided like any other and would pay in the file; a repeat whose count depends on the world
  stays with the frame.
- **Connections fan out over the apex within reach.** R31 connects every uncovered activation to everything on
  the apex of every frame it is open through, within its reach in every other dimension (D25). On a dense layout
  with little compression that is every point of the box, per open activation, and the box grows with the span
  (D4). It has not been measured, and under a policy whose future table reaches actions alone (D50) it never
  exists. Limiting connections to the highest levels, or to apex activations above the base, is the
  fallback, and it is not decided. An expected event that recreates the situation that inferred it is a loop with
  no world in it, and nothing breaks it but reward.
- **A plan spanning channels is never priced per channel.** A scoped reward reaches base actions of its channels
  and nothing else (R33), so a pattern whose call runs actions in two channels takes no scoped reward: its
  connections are priced only by unscoped rewards. A machine that pays per channel and lets patterns be
  connected to has no estimate for such a plan beyond what its base actions hold.
- **A call inferred by several situations** is placed at each of their coordinates plus the connection's offset
  (R36) and resolved per position (R47): in a channel with time alone those placements fall on one position and
  are one proposal several times, and in a channel with a layout they fall on several. No worked case has an
  action with a layout.
- **The hippocampus document predates D41.** Its moment is written as a class neuron that fires wherever a
  member fires (H2, H5). Under D41 a class is a variable with no neuron of its own, and what fires is a value
  neuron for what stood in it (D45), so the moment has to be restated.
- **The focus is the environment's.** A channel that needs one must provide it, as events the machine sees and
  base actions that move it.

- **What is said is chosen by share, never by reward.** The answer the copy and equal cases say is an expected
  event, resolved by share of voters, and an event dimension never reads an estimate (R47). Reward moves whether
  `say` runs and never which digit is said: a wrong digit with more voters is said for as long as those voters
  stand. That is right for an answer a teacher's lessons fix, and it means an answer that has to be chosen by
  what pays cannot be given as an expected event. A limit to come back to, not a fear.
- **The reward span is untested.** Every worked case pays each frame's action in that frame, stocks included,
  so R33's linear dissipation and its longest-open-window default have no case behind them. **Diagnostic:** a
  world that pays after a delay, with the span named and with it left to the default.

**Neighborhood size above the base.** A neighborhood is the frontier within reach (D5), and the reach is the
neuron's span plus `R` (D4). Among neighbors of about its own size that is a handful in each direction; among finer
ones it is many, and the alphabet a neighbor is chosen from grows as neurons are created. Nothing holds `|O|` fixed.
**Diagnostic:** neighbors per activation, against span.

**Parallelism.** The per-neuron passes are independent across neurons and could run at once. Re-centering makes
them slightly less independent, and D11 makes the neuron population smaller and each neuron busier — every
position sharing a type folds into one table, so the parallelism available shifts from across-neuron toward
across-activation, and re-centering becomes the contended point. The election is not sequential: R24 is two
decisions and a settling, each over every activation or every bid at once, with nothing revisited. The only ordering
constraint is that a level's bills and offers must all be in before its election runs, and both fall inside
one frame (T14). With nothing reported back, no bill waits on an election. On much larger inputs than MNIST all
of this needs revisiting.

**Asymmetric reach.** Backward reach emerges from the vote, and the forward window is
the same reach (D9). Whether one reach is right — "how much do I need to recognize myself" and "how far ahead do
I need to act" are different questions — is unresolved, and the connections are the whole of what the machine
does with the forward half. One reach is the committed choice; separate reaches are the fallback if diagnostics
show patterns consistently reaching the bound in one direction only. Across activation dimensions nothing is
assumed: each has the reach its own span gives it (D4).

**Whether a coarse voter should count as one.** The vote at the base gives every apex activation one vote per
dimension whatever level it stands at (R47). A level-4 neuron whose program expanded to forty base actions and a
base neuron that infers one action are then equal voters in the dimension they share, and the level-4 neuron's
estimate rests on a situation the base neuron's does not. Level was taken out of the vote because ranking by
level let compression override reward; whether some other reading of the voter — the strength behind its connection,
the exposures its estimate rests on — should weight it is not decided. **Diagnostic:** per dimension, how often the
winner was placed by a base voter over a pattern voter that disagreed, and which paid.

**The cross-neuron seam.** D27's abstention teaches a pattern not to name what another pattern of the *same
cover* holds. Across neurons nothing teaches it: a neighbor some other neuron's child reliably covers is still
present, still pays for its place, and stays in the pattern, paying its neighbor in the line. This is the same
omission as the neuron never hearing what it sold, seen from the pattern's side rather than the bid's,
and it is the same bet. The one-bit fallback above would not close it; closing it takes the per-neighbor report
the design removed. **Diagnostic:** per pattern, the share of its named neighbors that its bought bids were
never credited for, over its life.

**R33's shaping is ahead of the implementation; the cycle and the arithmetic are not.** The spec says a reward
carries an optional channel set and an optional frame span, dissipates linearly over that span, and enters the
estimate of the connection at the age each distance names. The current brain runs the two-frame cycle
as R29 says — the action inferred at one frame fires at the next with its reward attached — and folds rewards
into the estimate exactly as R31 says, one exposure, one share, weighted `1 / strength`. What it does not do is
shape them: it attaches the frame's reward whole to the action that ran in that frame, on every open age, which
is R33's span of one, and a minted pattern is pre-wired with `rewards[age − distance]` on a second path.
Neither scopes and neither ramps. **Diagnostic:** with a span-of-one reward, whether the estimate of an action
that ran in an earlier frame moves at all — under R33 it must not.

**The forward side of the code is the old model, and two of its pieces are the design.** The code's temporal
connections run toward every neuron that fired, event and action alike, at exact distances; it predicts events
from them, scores the predictions, and mints on the misses. None of that is the design: the design keeps action
connections only (D25), scores nothing forward, and mints on the bill alone (R14). What the code already does as
the design does: a connection is a lifetime total on the neuron, strengthened on observation, never weakened,
never collapsed, with the estimate the exact running mean and the walk wiring the next untried action on a
negative mean (R31, R37); and the vote at the base normalizes each voter to one vote per dimension and takes
actions by the share-weighted mean estimate, with no level in it (R47). What differs beyond the cut: the code
wires a declared default at birth at strength 1 and runs it where nothing infers, where the design has no
default and outputs nothing there (R35); the code connects to
base actions instead of the apex (D25), and so needs no expansion (R28, R30); covered activations keep learning
in the code, where D10 stops them the frame after they are covered; the code places a recognized pattern's
activation back at the frame its age names, where the spec places each child at the latest frame it covered
and the center of its box in space, and each value neuron where its value stood (D53); and it
shapes no reward (above).
[algorithm-implementation.md](algorithm-implementation.md) lists the changes.
