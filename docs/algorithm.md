# Universal Compression with Actions and Rewards (UCAR)

UCAR is the design for the full pattern lifecycle: recognition of existing patterns, creation of new ones, their
refinement after birth, and their structural revision — reuse, defragmentation, splitting. The name states the
theory: the substrate is an online compression engine whose dictionary entries are situations ("universal" in the
coding sense — KT estimation and MDL pricing are universal-coding machinery, tuned to no assumed source), and
actions and rewards ride on that dictionary as connection payload — the compressor decides what exists, reward
decorates what it is worth, and salience-driven one-shot memory is explicitly out of scope (that is a hippocampal
function; this is cortex).

The mechanisms are specified on the spatial axis (same-frame co-occurrence, distance 0) and generalize to the
temporal axis by swapping the co-occurrence counter — see [Temporal generalization](#temporal-generalization).
The neuron-side machinery lives in [`brain/brain-core/src/neuron.rs`](../brain/brain-core/src/neuron.rs) —
recognition under the `match_info` flag, creation under `error_info` — and the thalamus-side machinery in
[`brain/brain-core/src/thalamus.rs`](../brain/brain-core/src/thalamus.rs).

## The currency: two-part code

The pattern set is a code for the brain's experience. Its total cost has two parts:

```
L(total) = L(model)   bits to store the patterns: names, context entries, connections
         + L(data)    bits to encode what is observed each frame, given the patterns
```

Prediction and compression are the same quantity: a model that predicts well makes observed frames cheap to encode.
The conversion between probability and bits is fixed — an outcome the model assigns probability `p` costs
`-log2(p)` bits. Confident correct predictions are nearly free; confident wrong ones are expensive; a coin flip
costs its full bit. This is a proper scoring rule: no model profits from overclaiming or underclaiming.

Both parts have physical carriers in the substrate:

- **Model bits live in references.** A context entry or connection is a pointer that singles out one neuron among
  the candidates it could have named. Distinguishing one among `N` costs `log2(N)` bits, where `N` is the
  *addressable* population — the alphabet the reference actually chooses from — not the whole brain.
- **Data bits live in activations.** A neuron's firing event carries its own surprisal: a child that fires on one
  frame in a thousand delivers ~10 bits when it fires. The competition among a parent's children is a questionnaire
  and the winner's identity is the answer; the sparse high-level active set is the compressed encoding of the frame.
  Nothing stores this stream — prediction error is its per-frame increment, and the ledgers below meter it.

Every structural decision — create, reuse, merge, split — is one question: does the move reduce `L(total)`?
Accept iff the change is negative. No similarity thresholds exist anywhere in the design.

### Scope: the ledgers are deliberately local

Each neuron keeps its own ledger over its own neighborhood, and the sum of local ledgers is not the brain's
description length — the same base event is witnessed, and billed, by every parent whose neighborhood contains it.
This is accepted: local ledgers gate local decisions cheaply and in parallel, which is what a distributed substrate
requires. The distortion the double-billing causes — many parents independently paying for the same pattern — is
corrected structurally, at the one point where requests converge: the thalamus reuse gate below. Detection is
local; identity is global; nothing else needs a global view.

### Horizon: rent is the window

`L(data)` accrues per frame, so every economic comparison needs a horizon. The horizon enters the design in
exactly one place: **rent** — every piece of structure pays its own storage price amortized over the horizon (see
[Forgetting](#forgetting-earn-rent-death)), so "worth its price" always means "worth its price over the window the
rent defines." Structure must persist, not merely occur, to justify itself — the same bet birth already makes.

Deriving an optimum horizon from context length is a planned experiment, not a precondition: every mechanism here
consumes the horizon identically wherever it comes from.

## The economic frame: online facility location

Pattern creation is an instance of online facility location, run in the deterministic accumulate-then-open form:

- **Demand points** arrive one at a time: prediction failures, each carrying an observed context and a benefit
  measured in bits.
- **Opening a facility** — creating a pattern — costs a fixed fee: the pattern's storage price, also in bits.
- Demand accumulates at cluster centers; a center opens its facility once the accumulated benefit covers the fee.

The number of patterns is not an input anywhere. It emerges from the economics: tight recurring context clumps pay
for their own facility, scattered noise never does. The one quantity that looks like a threshold — the opening
fee — is the MDL storage price, not a tuned constant.

One design judgment sits on top of the facility-location skeleton and is stated plainly: assignment clusters on the
*context* axis (when the failure happens) while deposits measure the *target* axis (what failed, in bits of
surprise). Clustering on when, paying with what. The formal competitive-ratio guarantees for online facility
location cover the accumulate-then-open skeleton, not this metric split, nor evidence decay or center drift.

## Estimation: KT for models, raw MLE for background

Every likelihood-ratio score in the design compares a model's per-entry probability against a background's. The two
sides are estimated differently, on purpose:

- **Model side (pattern contexts, embryo centers): Krichevsky–Trofimov.**

  ```
  p_c = (count + 1/2) / (n + 1)
  ```

  KT is the minimax-regret universal estimator — the MDL-native answer to small samples, derived rather than tuned.
  It keeps probabilities off the 0/1 boundary without an arbitrary cap. The effect is at the cold start: a fresh
  center's entries score as ~0.75-probability members instead of certainties, so a near-miss second occurrence
  receives a mild penalty instead of ~10 bits per absent entry, and fuzzy pooling works in the regime where
  boundaries are actually drawn.

- **Background side: raw MLE with the skip rule.** Background frequencies are plain counts over frames, unsmoothed.
  An outcome the background has never witnessed cannot be priced against it, so its term is skipped rather than
  smoothed into a finite surprise: a present entry with `p_bg = 0`, or an absent entry whose neighbor has co-fired
  on every frame so far (`p_bg = 1`), contributes nothing. The background is the null hypothesis; inventing mass
  for it would price surprises against events that never happened.

Each neuron keeps two background models: a **context** model (how often each same-level neighbor is co-active,
per context frame) feeding recognition and embryo assignment, and an **inference** model (how often each base event
is active in the inference neighborhood, per inference frame) feeding the benefit gate. Both are learning-gated
like every substrate update, so frozen evaluation stays frozen.

## Recognition

Candidate child patterns are found via the same-level context index — patterns sharing at least one context neuron
with the observed co-activation. Each candidate is scored by log-likelihood ratio — its own context model against
the neuron's background model — and fires iff its rank is positive; the highest positive rank wins, and at most one
pattern fires per parent per frame. No similarity threshold exists anywhere in the decision.

For each context entry `e` the candidate stores with strength `s`, over `n_c` trials (the candidate's lifetime
fire count — immortal and monotonic, so per-entry probabilities never exceed 1; the model stays adaptive through
refinement's mode collapse, not through decay):

```
e present:  contributes log2( p_c(e) / p_bg(e) )
e absent:   contributes log2( (1 - p_c(e)) / (1 - p_bg(e)) )
p_c(e)  = (s + 1/2) / (n_c + 1)                    // KT
p_bg(e) = context count of e / context frames      // raw MLE, skip rule at 0 and 1
```

Observed neighbors the candidate does not store contribute nothing: the pattern's model has no opinion on them and
the background explains them at its own rate on both sides of the ratio.

On a fire, matched entries strengthen (`s += 1`) and the pattern's trial count increments — one event, one
increment; the rank is banked into the pattern's balance (see [Forgetting](#forgetting-earn-rent-death)). This per-fire strengthening is also the refinement mechanism (see
[Context refinement](#context-refinement-plasticity-of-degree-one)).

## The surprise gate: is this failure worth anything?

A neuron whose child fired this frame is *subsumed* — it casts no prediction and evaluates nothing, since the child
already represents it. Only a neuron with no firing child predicts and gets evaluated.

When no child fired, the neuron casts its own prediction from its connections and compares it against the observed
base events. A divergence is scored as net evidence for a distinct source, all terms in surprisal bits against the
neuron's inference background model `p_bg(e)`:

```
evidence_for     = Σ_{present, unpredicted}  -log2(p_bg(e))        // bits a correct prediction would have saved
                 + Σ_{predicted, absent}     -log2(1 - p_bg(e))    // a reliable event vanishing, priced by its reliability
evidence_against = Σ_{predicted AND present} -log2(p_bg(e))        // bits the existing mechanism already saves
benefit = evidence_for - evidence_against
```

Hits vary independently of misses, so the net score can genuinely go negative: a frame with a large miss still nets
out as no surprise when the same prediction nailed rarer events. The fourth case — correctly unpredicted and
absent — is uninformative under either hypothesis and is not scored. A neuron with zero inference frames has no
background model and scores every failure at zero.

`benefit <= 0` drops the failure entirely: an ordinary mismatch on an otherwise-sharp prediction is noise, not
signal. A positive benefit becomes a demand point for the womb.

## The womb: embryos as cluster centers

Each parent neuron holds a womb: a small set of **embryos**, the not-yet-born cluster centers. An embryo is
context-only and owns no neuron, no connections, and no target data:

- **context center** — per-neighbor count plus an occurrence total `n`; `count / n` is a soft membership
  distribution.
- **evidence** — accumulated benefit deposited by the failures assigned to it, in bits.

The child's connections are deliberately absent from the womb. From the parent's perspective a child's connections
cost the parent nothing — they belong to the child once it exists, are seeded at birth from the triggering frame,
and are refined by the child's own connection learning afterward. Pricing or clustering on target data would charge
the parent for storage it never holds.

### Assignment: serve-or-open

A failure's observed context is scored against every embryo's center with the same likelihood-ratio math
recognition uses on born patterns — an embryo scores exactly like a routing entry whose strengths are its center
counts, KT-estimated over `n`. Distance and match quality are one number: the rank.

- **Best embryo with rank > 0**: the failure is served by that embryo. Its context counts fold in (`count += 1` per
  observed neighbor, `n += 1`) and its evidence gains this failure's benefit. The center drifts toward the failures
  actually assigned to it — an online mean, no learning rate.
- **No embryo with positive rank**: the failure opens a new embryo seeded from its own context, with its benefit as
  opening evidence. Opening never runs the birth check — a pattern is only born when a later failure serves the
  embryo and its pooled evidence covers the price.

Serving is fuzzy by construction: near-miss contexts pool into the same embryo because the rank is a likelihood
ratio, not an exact-match test. This is what keeps womb size bounded — one embryo per recurring context cluster,
not one entry per unique context ever observed.

### Price

An embryo's price is what the parent will actually store, in bits:

```
price = name_bits(|center|, alphabet) + log2(children + 1)

name_bits(k, n) = log2( C(n, k) )  =  Σ_{i=0}^{k-1} log2( (n-i) / (k-i) )

alphabet = max(context neighbors ever witnessed, center size)
children = the parent's current child pattern count
```

The center is named as what it physically is — an unordered subset of the alphabet this parent has actually seen —
and the pattern's own name distinguishes it among the parent's children. Naming a set, not `|center|` independent
pointers: the pointer form can spell the same set in `k!` orders (wasting `log2(k!)` bits) and overcharges dense
centers badly — naming 25 members of a 30-symbol alphabet is the same information as naming the 5 absentees, and
naming all `n` of `n` carries zero information and costs zero here. Both terms are measured from local state the
neuron already tracks — no global population figures. Consequences worth stating plainly: prices scale with
neighborhood richness, so patterns in dense contexts must earn more evidence; and degenerate early alphabets yield
near-zero prices, which is the honest MDL statement — names drawn from a tiny alphabet genuinely cost almost
nothing to store. No floor is imposed.

### Embryo death

Deposits are strictly positive (the surprise gate drops non-positive benefits before assignment), so embryo evidence
is only ever driven down by rent — the embryo's price amortized over the horizon, the same rule everything pays
(see [Forgetting](#forgetting-earn-rent-death)). Eviction happens when effective evidence reaches zero. There is no
separate incoherence mechanism; an embryo fed poorly is an embryo whose deposits arrive too slowly to outrun rent,
and it dies the same way.

## Birth: the converged center, actually

Birth happens when a served embryo's accumulated evidence covers its price. A pattern neuron is created one level
above the parent's own level, and the child inherits the embryo's distribution, not its member set:

- each context entry is installed at strength = its center count;
- the child's recognition trial count starts at the embryo's occurrence total `n`;
- the child's opening balance is the evidence the embryo accumulated — infancy is pre-paid by the womb.

So the newborn's likelihood model starts at `p_c ≈ count / n` per entry — the soft membership the womb converged
to. A neighbor served once in fifty enters at weight 1/50, its absence costs ~0 bits, and it dilutes away through
ordinary refinement instead of rendering the pattern unfireable. Center shedding needs no explicit prune rule:
low-share members are born low and fade. A pattern born from more evidence starts further from death — the same
evidence buying the same durability (see [Forgetting](#forgetting-earn-rent-death)).

Founding connections are wired from the birth frame's observed base events (id, channel, dimension, reward), so the
child predicts from its first activation and its own connection learning refines from there. Patterns represent
situations; their initial reaction to the situation may be poor, and learning it is the child's job, not grounds
for death. Born patterns die one way: the situation stops recurring and activation decays away.

## Context refinement: plasticity of degree one

Refinement is the counting form recognition already implements: on a fire, matched entries strengthen and absent
entries dilute relatively as the trial count grows. This is an online mode collapse — the context converges to the
recurring core of the frames the pattern actually fires on. Two properties matter:

- **Boundary repair.** A situation whose boundary was drawn slightly wrong at birth converges to the right boundary
  through ordinary fires. Structure below the level of "which patterns exist" self-corrects continuously.
- **Narrow-then-remint.** A pattern pooling two situations sharpens toward the dominant one; its rank on the
  minority mode eventually goes negative and it stops firing there. Subsumption releases those frames, the parent's
  womb re-accumulates, and the minority situation mints separately. Refinement is not a split operator, but it is
  an eviction operator that lets the womb do the splitting — slowly, at the cost of re-paying the price.

The hard add/remove form of refinement — snapping the stored context toward each fired frame — is rejected: it
tracks the most recent frame instead of converging to a mode, and churns under alternating situations.

## Connection refinement: the reaction is always plastic

A pattern's context says when it fires; its connections say what it predicts when it does. The connections are
refined by ordinary connection learning on every frame the pattern is active: edges toward observed co-actives
strengthen, rewards update by exponential smoothing, and prediction is the position-winner competition — the
strongest bucket per position wins, so the effective prediction is the modal configuration of the frames the
pattern actually fires on. Connections never decay in absolute strength; forgetting is competitive and economic:

- **Contested positions displace.** When the frames a pattern fires on shift, the newly recurring targets
  out-strengthen the stale ones and win their positions. The old prediction is outvoted, not erased.
- **Uncontested residue goes economically weightless.** A stale target nothing displaces keeps being predicted and
  keeps missing — but each miss is priced at `-log2(1 - p_bg)`, and as the stale event stops appearing in the
  pattern's inference neighborhood its background frequency dilutes toward zero, so the miss price decays toward
  zero bits. The background model forgets on the ledger side even though the synapse persists; the residue costs
  memory, never economics.

This is also the whole death semantics of multi-parent children. A pattern with parents A and B error-corrects X
for A and Y for B. When A's routing entry decays out and A releases the pattern, it continues to infer X at first;
but it now fires only on B's frames, so refinement drifts its connections toward Y, X's targets lose their
contested positions, and X's uncontested remainder dilutes into weightlessness. No explicit weakening rule exists
or is needed — release-on-last for the pattern's existence, and refinement plus background dilution for everything
it once predicted.

## Forgetting: earn, rent, death

In a stationary world an MDL compressor never forgets — structure that paid for itself once keeps paying, because
its situations keep recurring. Forgetting is purely a consequence of non-stationarity: when the source drifts, a
pattern whose situation stopped recurring still costs its model bits while saving no data bits, and deleting it
strictly reduces `L(total)`. Forgetting is the compression objective applied over a moving window, implemented as
one economic rule for everything:

- **Earn.** When a pattern fires, its recognition rank is the log-likelihood ratio of the observed context under
  its model versus background — literally the data bits the pattern saved on that frame compared to coding the
  frame without it. The rank is banked into the pattern's **balance**. It is already computed on every fire;
  banking it is free.
- **Rent.** Every piece of structure pays its own storage price amortized over the horizon:
  `rent = price / horizon` per frame. A big blurry pattern owes more per frame than a small crisp one
  automatically, because its price is bigger.
- **Death.** Balance reaching zero releases the structure. One rule for every lifecycle stage: an embryo's
  deposits are its earnings and its eviction is insolvency; a newborn's opening balance is the evidence its
  embryo accumulated; a multi-parent child keeps a balance per parent routing entry — each entry banks the rank
  earned in that parent's tournament, pays rent on that parent's storage, and dies alone, with release-on-last
  unchanged.

Survival therefore rewards bits saved — sharpness times recurrence — not fire frequency. A broad mush pattern
firing often at ~0.3 bits a fire must fire twenty-five times as often as a sharp pattern earning 8 bits to stay
solvent: breadth is not fitness. This is the post-birth accountability that pure activation counting cannot
provide, priced in the same units as everything else.

The accounts are separated: the balance is economic state, and the likelihood model's trial count is statistical
state. Trials are the lifetime fire count — immortal and monotonic, so per-entry probabilities never exceed 1 —
and the model stays adaptive through refinement's mode collapse, not through decay. The horizon in the rent is the
one place a time constant survives in the design.

The death ledger keeps its role as scheduler: on every bank event, remaining life is `balance / rent`, and the
death frame is re-registered from it. Death frames are never persisted — they are recomputed from materialized
balances on restore.

## Structural plasticity

### The signal division

Co-occurrence and identity are different signals with different owners:

- **Co-occurrence belongs to the hierarchy.** Two patterns persistently co-firing is same-level context for the
  level above — food for the womb, which builds a conjunction pattern on top. This is the existing machinery and
  nothing else consumes its signal.
- **Identity belongs to the thalamus.** Two patterns being the *same thing* — near-identical connection sets,
  interchangeable contexts — is invisible to any neuron and useless to the hierarchy. Building a conjunction over
  two copies of one thing represents an identity as a structure, which is the pathology behind apex explosion.

No central firing statistics are kept. There is no co-fire matrix. Every structural check below is attached to an
event that already flows through the system, in the style of online dictionary compressors: invariants enforced at
the moment a count changes, never by a sweep. (Sequitur mints a rule the instant a digram repeats and dissolves it
the instant its use count drops; adaptive Huffman restructures its whole codebook one local swap per increment; CTW
never commits at all, carrying both merge and split hypotheses as weights — with KT as its leaf estimator.)

### Merge = reuse, decided at birth-request time

A birth request arriving at the thalamus is a demand point, and existing same-level patterns are the facilities.
Serve-or-open, one level up from the womb:

```mermaid
flowchart TD
    A[Birth request: converged center + target events] --> B{Existing same-level pattern<br/>explains this request?}
    B -->|serve: reuse is cheaper| C[Install existing pattern into the requesting parent's<br/>routing table, with the embryo's center as its local context]
    B -->|open: nothing serves| D[Mint a new pattern neuron]
    E[Same frame, multiple requests] --> F[Pool requests with shared targets first,<br/>then serve-or-open once per pool]
```

The serve test is the same MDL question in the same units: serving costs one routing entry in the requesting parent
(`|center| × log2(alphabet)` bits, which the parent pays either way) plus whatever the existing pattern's
connection model loses by covering a second parent's frames; opening costs a full new pattern — name, connections,
everything the request would create. For genuine duplicates — the common case, since overlapping neighborhoods
witness the same situation — serving wins by roughly the entire cost of the duplicate.

Candidate lookup is by target signature: patterns predicting substantially the same base events as the request's
target events. Requests born in the same frame are pooled before lookup — parents witnessing one surprise from
adjacent neighborhoods produce requests with heavily overlapping targets, and one child serves the whole pool.

A served pattern becomes a **multi-parent child**: one connection set, one id, and per-parent routing entries, each
with its own context view, its own balance, and its own death clock. The pattern dies in a parent when
that parent's entry decays out, and dies entirely when its last parent releases it. What it predicted for a parent
that released it is forgotten indirectly, through connection refinement — see
[Connection refinement](#connection-refinement-the-reaction-is-always-plastic).

### Defragmentation: the runner-up trigger

Two children of one parent cannot co-fire — recognition fires at most one — so duplication inside a parent shows up
as interchangeability: whenever A wins, B is the perennial near-rank runner-up. The parent sees this in the
tournament it already runs, for free. A persistent runner-up (tracked as a decayed count on the routing entry) is a
merge proposal sent up with the frame's results. The thalamus applies the standard test: the pooled model codes the
union of both children's frames at some cost; if the fragments were one situation, the pooled probabilities are
*more* accurate (each fragment had overfit its half) and the merge saves both data bits and a pattern's storage; if
they were genuinely different, the pooled model is broader, codes every frame worse, and the merge rejects itself.

### Split: the post-natal womb

A pattern's own context counts cannot reveal that its frames are bimodal — marginals discard the correlation that
distinguishes one noisy situation from two clean ones. The statistic that preserves mode structure is the womb's
own machinery, reused: each born pattern carries a small set of **sub-centers**, and every frame it fires, the
observed context is served into them by the same serve-or-open likelihood-ratio assignment embryos use. A unimodal
pattern's matches all serve one sub-center; a bimodal pattern's matches sort themselves into two.

A split is proposed when the second sub-center's accumulated evidence covers the *incremental* price of the split:
one new pattern name plus the entries by which the two tightened centers exceed the original. The data side of the
test includes the chooser term — with two patterns, each frame must also encode which one fired, ~1 bit — which is
what makes gratuitous splits permanently unprofitable: splitting a unimodal pattern buys zero data bits and pays
storage plus a bit per frame forever. Split and merge of the same pair evaluate the same difference with opposite
signs, so an accepted split implies the reverse merge rejects: the system cannot oscillate.

On an accepted split, the thalamus executes: two children seeded from the two sub-centers' converged counts (the
same birth-seeding rule), the original's balance divided by each sub-center's share, the original
released. Detection is entirely local; the thalamus only allocates ids and wires.

## One frame, in order

```mermaid
flowchart TD
    A[Observed same-level context] --> B{Recognition: does a born child fire?}
    B -->|fires| C[Subsumed — child's sub-centers updated,<br/>matched entries strengthen]
    C --> D{Second sub-center covers<br/>incremental split price?}
    D -->|yes| E[Split request to thalamus]
    B -->|no fire| F[Cast prediction, compare against reality]
    F --> G{Surprise gate: net evidence > 0 bits?}
    G -->|no| H[Drop — noise]
    G -->|yes| I[Serve-or-open into the womb]
    I --> J{Embryo evidence ≥ price in bits?}
    J -->|yes| K[Birth request to thalamus]
    K --> L{Thalamus: existing pattern serves?}
    L -->|serve| M[Reuse — install into parent]
    L -->|open| N[Mint new pattern]
```

## Temporal generalization

Nothing above is same-frame-specific except which counter supplies co-occurrence. Spatial context counts answer
"how often is this neighbor active in the same frame as the parent"; temporal context counts answer the same
question per distance `d`. The currency, KT estimation, the womb, the price, the refinement dynamics, and the
structural moves carry over with the distance axis as a parameter of the background model, not a separate
mechanism. The temporal side currently runs a separate adaptive-grouping mechanism and keeps its own parallel
structures; nothing here unifies them, and porting is deliberately deferred until the spatial side is proven.

## Implementation plan

Phased so each lands independently and is measured before the next. The standing metrics per phase: MNIST accuracy,
neuron counts per level, apex pattern count, and wall-clock per frame.

**Phase 1 — currency (done).** `compute_spatial_evidence` in surprisal bits with skip rules and a zero-frames
guard; womb price as `|center| × log2(alphabet) + log2(children + 1)`. Minting rates shift under the new units;
this phase is measured before anything else changes. Note: snapshots taken before this phase carry womb evidence in
linear units; those wombs decay away harmlessly, but cross-phase snapshots should not be used for measurements.

**Phase 2 — estimation and birth seeding.** KT estimator in embryo assignment and pattern recognition ranking,
removing the 0.999 cap. Birth seeding: context entry trial counts from center counts and occurrence total, and the
embryo's evidence as the newborn's opening balance (plumbed through the correction request and install op, which
currently carry only the member list). Expected effects: fewer fragmented embryos, newborns that fire and refine,
lower dead-on-arrival pattern counts.

**Phase 3 — economic forgetting.** Separate the two accounts on routing entries: trials become the immortal fire
count feeding the likelihood model; a new balance field banks the recognition rank on every accepted fire and pays
rent (`price / horizon`, with price carried from birth). Candidacy and death read the balance; the embryo's
flat-rate staleness decay becomes the same rent on its own price; the death ledger re-registers from
`balance / rent` on each bank event, recomputed from materialized balances on restore, never persisted. Expected
effects: broad low-rank patterns die insolvent, sharp rarely-firing patterns survive; measured by pattern churn,
per-level counts, and accuracy.

**Phase 4 — thalamus reuse.** The largest phase, gated on a plumbing audit first: per-parent routing entries
already exist, but subsumption, the death ledger, and the delete cascade assume one parent per pattern, and
"dead in one parent, alive in another" needs an owner (per-entry balances from phase 3 provide it). Then: a
target-signature index over existing patterns (maintained at mint/death, queried at request time), same-frame
request pooling by target overlap, the MDL serve test, and an install-into-parent op for served requests. Measured
primarily by apex pattern count and per-level neuron counts — this is the phase aimed at the apex explosion.

**Phase 5 — post-natal womb and split.** Sub-centers on born patterns (the embryo struct reused, serialized with
the routing entry), served on each fire; split request when the second sub-center covers the incremental price;
thalamus-side execution with the original's balance divided by each sub-center's share. Measured on deliberately
pooled synthetic situations — two interleaved contexts with distinct targets — for split latency and correctness,
then on MNIST.

**Phase 6 — defragmentation.** Decayed runner-up counts on routing entries, merge proposals over a persistence
line derived from the same economics (the pooled-versus-separate coding test), thalamus-side execution. Last
because phases 2 and 4 should prevent most fragmentation from arising; this phase collects what still slips
through.

## Open questions

- **Serve-test tractability.** The reuse decision compares coding costs that depend on the existing pattern's fit
  to the requesting parent's frames, which the thalamus cannot observe directly. The practical proxy — target-set
  overlap between the request and the candidate — needs validation against cases where targets overlap but
  contexts genuinely differ.
