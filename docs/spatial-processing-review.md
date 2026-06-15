# Spatial Processing — Review Findings and Required Changes

Review of the `spatial` branch at commit `1e1b460` ("spatial processing - continuing progress"),
checked against the design in [spatial-processing.md](spatial-processing.md) and against the
intended algorithm semantics. All 88 brain-core tests pass at this commit; the issues below are
behavioral, not compile/test failures.

Two kinds of findings are listed: **bugs** (code does something wrong relative to any reasonable
reading) and **translation gaps** (code is internally consistent but does not implement the
intended algorithm — these are the reason the spatial hierarchy doesn't behave as designed on
MNIST).

---

## Status (updated)

The dependency chain **1.1 → 2.1 → 2.2 has landed** and is verified in the current `spatial` branch
— these are what produced the 95.73% full-MNIST result logged in
[mnist-spatial-experiments.md](mnist-spatial-experiments.md). Confirmed in code:

- **1.1 — observed-context neighbor filtering: DONE.** `dispatch_spatial_frame` builds a per-task
  observed `SpatialContext` filtered to the parent's neighbor channels
  ([thalamus.rs:1461](../brain/brain-core/src/thalamus.rs)).
- **2.1 — per-position winner competition: DONE.** `mint_spatial_corrections` reduces the d=0 votes
  to one winner per `(channel, dim)` via the `position_winners` map before scoring error
  ([thalamus.rs:1101](../brain/brain-core/src/thalamus.rs)).
- **2.2 — coordinate inheritance for minted patterns: DONE.** `allocate_spatial_pattern_neuron`
  inherits the parent's full coordinate and deliberately does **not** register it in
  `neurons_by_value` ([thalamus.rs:468](../brain/brain-core/src/thalamus.rs)).

**Still open and scheduled** (see [roadmap.md](roadmap.md)):

- **1.2 — snapshot restore** — scheduled under the roadmap's *Backups / imports / exports (Phase 4)*
  item.
- **Minor — committed `*.node` binaries** — scheduled under the roadmap's *Do not commit platform
  binaries* item.

A **new open question** surfaced from 2.2: *should temporal corrections inherit channels the same way
spatial corrections now do?* Tracked as its own roadmap experiment.

The original findings are retained below for the record; the headings carry their current status.

---

## 1. Bugs

### 1.1 Spatial pattern matching can never succeed when neighborhoods are declared — HIGH — ✅ DONE

**Where:**
- Observed context built unfiltered: `get_spatial_level_tasks` ([thalamus.rs:1203-1237](../brain/brain-core/src/thalamus.rs)) puts **every** neuron active at the level into one shared `SpatialContext`.
- That global context is handed to every neuron's `recognize_spatial_patterns` ([neuron.rs:1040-1101](../brain/brain-core/src/neuron.rs)) and scored by `SpatialContext::match_observed` ([context.rs:85-125](../brain/brain-core/src/context.rs)).
- Stored pattern contexts, by contrast, are **neighbor-filtered** at mint time ([thalamus.rs:1051-1062](../brain/brain-core/src/thalamus.rs)) — the parent's neighborhood only.

**Why it's fatal:** `match_observed` counts every observed neuron absent from the known context
as *novel* and gates on Jaccard `common / (common + missing + novel) ≥ merge_threshold`. On 7×7
binary MNIST with radius-1 neighborhoods, every pixel channel fires one bucket per frame, so the
observed set is ~49 entries while a stored context is ≤ 8. The ratio tops out around 8/49 ≈ 0.16 —
below every threshold the jobs probe (0.5 / 0.7 / 0.9 in `threshold-grid.js`). Routing matches
essentially never fire.

**Knock-on effects:**
- The spatial hierarchy never activates: no subsumption, apex = all sensory pixels, temporal
  never sees groupings.
- **Unbounded duplicate minting.** Dedup relies entirely on a minted correction matching next
  time (match → vote suppression → no error votes → no re-mint; see the `predicted_events`
  gate at [thalamus.rs:1080](../brain/brain-core/src/thalamus.rs)). Since matching can't pass,
  the same parent re-errors and re-mints an identical correction on every exposure.

**Fix:** filter the observed context per parent to the parent's neighbor channels before
matching — same treatment the dispatch already applies to `task_actives`
([thalamus.rs:1410-1417](../brain/brain-core/src/thalamus.rs)) and the mint pass applies to
error evaluation ([thalamus.rs:1042-1077](../brain/brain-core/src/thalamus.rs)). Either build a
per-task filtered context, or make the novel pass in `SpatialContext::match_observed` ignore
non-neighbor entries.

Note this only bites when neighborhoods are declared; all-pairs domains (stocks, text) have
symmetric stored/observed sets and are unaffected.

### 1.2 Snapshot restore silently destroys spatial structure — HIGH — ⏳ OPEN (roadmap: Phase 4 backups)

Serialization was updated for spatial but restore was not, so a save/load round-trip corrupts
silently instead of failing loudly:

- **Spatial levels are dropped.** `SnapshotNeuronEntry` carries only the temporal `level`
  ([thalamus.rs:174-179](../brain/brain-core/src/thalamus.rs)); `get_snapshot`
  ([thalamus.rs:1851-1871](../brain/brain-core/src/thalamus.rs)) never reads
  `neuron_spatial_levels`, and `restore_snapshot` clears the map via `reset()`
  ([thalamus.rs:1938](../brain/brain-core/src/thalamus.rs)) and never repopulates it. Every
  neuron comes back at `spatial_level = 0`.
- **Spatial routing tables land on the temporal side.** `serialize_children` and
  `serialize_context_refs` flatten spatial and temporal entries into one undiscriminated list
  ([neuron.rs:404-443](../brain/brain-core/src/neuron.rs); spatial contexts marked
  `distance: 0`). `load_neuron` ([column.rs:600-645](../brain/brain-core/src/column.rs)) restores
  through `add_child` / `add_context` / `add_context_ref` — the back-compat shims that route
  **unconditionally to the temporal structures**
  ([neuron.rs:944-954](../brain/brain-core/src/neuron.rs)). After restore, every spatial routing
  entry and context ref sits in the temporal tables with d=0 contexts: spatial hierarchy gone,
  temporal tables polluted.
- Connections and Welford error stats *do* round-trip correctly (`create_connection` routes
  d=0 to `spatial_connections`; `age == 0` stats route to the spatial bucket,
  [column.rs:634-642](../brain/brain-core/src/column.rs)) — which deepens the false sense of
  safety.

**Fix:** add a spatial/temporal discriminator to `SerializedChild` and `SerializedContextRef`
(or split them into separate fields), restore each to the proper side; add `spatial_level` to
`SnapshotNeuronEntry`, populate in `get_snapshot`, restore into `neuron_spatial_levels`.
Then implement the Phase 4 acceptance test from the design doc (train → snapshot → restore →
identical next frame), which would have caught all of this.

### 1.3 Temporal novel-detection guard accidentally relaxed — MEDIUM — ⏳ OPEN (roadmap: near-term fix)

`TemporalContext::match_observed`'s novel pass changed from `pattern_distance < 1` to `< 0`
([context.rs:270](../brain/brain-core/src/context.rs)), and the explanatory comment ("context
neurons must be older than the parent") was deleted. History: relaxed in `c2f06bf` when spatial
still shared the temporal matcher (the design doc anticipated relaxing the d<1 guard for
spatial); the split in `5c96075` gave spatial its own `SpatialContext::match_observed` but never
restored the temporal guard.

**Effect:** for parents matched at ages ≥ 1, same-frame co-active neurons not present anywhere
in the known context now count as novel mismatches (`has_partial_match` only rescues neurons
known at *some* distance), depressing the Jaccard ratio and failing temporal matches that passed
before the branch.

**Fix:** restore the guard to `< 1`.

---

## 2. Translation gaps — code does not implement the intended algorithm

### 2.1 No per-position competition in spatial prediction / error evaluation — ✅ DONE

**Intended:** each neuron predicts its 8 neighbor *positions*; for each position the bucket
predictions (e.g. black vs white) **compete and one wins**. The predicted co-activation set is
the modal local configuration — exactly one bucket per neighbor channel — and the error is "how
many positions deviated from what I expected."

**Implemented:** `generate_spatial_votes` casts **every** `connections[0]` entry as a vote
([neuron.rs:1106-1133](../brain/brain-core/src/neuron.rs)), and `mint_spatial_corrections`
compares that raw union symmetrically against the observed set
([thalamus.rs:1064-1097](../brain/brain-core/src/thalamus.rs)). No competition step exists
anywhere between voting and error scoring. (The temporal `evaluate_vote_error` has the same
no-competition shape, [thalamus.rs:1580-1614](../brain/brain-core/src/thalamus.rs) — the spatial
pass inherited it. The per-dimension competition that *does* exist in the codebase —
`determine_dimension_winners` in the consensus pipeline — is only applied to inference output,
never to error evaluation.)

**Why it degenerates:** the predicted set is a union over lifetime history. For binary pixels
with radius-1 neighborhoods, once a pixel has seen varied digits it is connected to **both**
buckets of all 8 neighbors: predicted = 16, observed = 8, missing = 8, novel = 0 → error rate =
**exactly 0.5**, which sits precisely on the default 0.5 threshold and stops minting forever
(`error_rate <= threshold` → skip). With b buckets the saturated rate is (b−1)/b — e.g. 0.9 for
10 buckets — *permanently above* any sane threshold, so minting never stops. Either silent or
incontinent; the error signal measures novelty-vs-everything-ever-seen, not deviation from the
likely configuration. This is the structural reason threshold tuning (the `threshold-grid` job)
can't find a good operating point.

**Fix:** in the spatial vote/error path, group the d=0 votes by the target neuron's channel
(+ dimension), take the strongest per group as the winner, and compare winners against the
observed bucket per channel. Predicted set becomes exactly one neuron per neighbor position;
error becomes a Hamming distance from the modal patch; minting quenches naturally once local
statistics stabilize.

### 2.2 Pattern neurons have no spatial topology — hierarchy jumps to whole-image fingerprints — ✅ DONE

**Intended:** an L1 correction continues to error-correct **the parent's neighborhood**.
Receptive fields grow gradually: L2 groups L1 patch-detectors that are spatial neighbors, etc.

**Implemented:** minted pattern neurons get no channel. `get_neuron_channel_id` returns `None`,
the `unwrap_or(0)` sentinel kicks in, and `is_neighbor_channel` falls through to **all-pairs**.
The comment at [thalamus.rs:245-256](../brain/brain-core/src/thalamus.rs) rationalizes this by
claiming pattern neurons "only ever co-fire with [neurons] bounded by their parent's neighbor
graph anyway" — that claim is false: one L1 can fire per pixel position, so the L1 fired set
spans the whole image. Consequences:

- An L1's d=0 connection learning and error evaluation run against the **entire frame**
  (`full_actives` filtered by a neighbor set that is all-pairs).
- When an L1 errors, the L2 it mints has as context **every L1 that fired that frame** — a
  whole-image fingerprint. The hierarchy goes patches → memorized images in one hop, instead of
  edges → strokes → parts.
- Because the full frame is never fully predictable, L1s error perpetually → one fresh L2 per
  novel image, forever (neuron growth scales with the training set).

**Fix:** minted corrections inherit the parent's full **(channel, dimension, coordinate)** — not
just the channel. `allocate_spatial_pattern_neuron` already receives `parent_id`
([thalamus.rs:1100](../brain/brain-core/src/thalamus.rs)), so inheritance is small. Effect: L1's
learning/error stays filtered to the parent's 3×3, and an L2 minted from L1-at-position-A gets
as context only L1s whose parents are A's neighbors — receptive fields grow one radius hop per
level. Inheriting the full coordinate (not the channel alone) is also what lets apex spatial
patterns participate in temporal processing without disturbing its channel-based consensus.
The design rationale, implementation constraints, and downstream consequences follow.

**Design decision (from design discussion):** when a spatial neuron gets a new child
(correction), the child inherits the parent's full **(channel, dimension, coordinate)**. This is
the mechanism by which apex spatial patterns participate in temporal processing without
disturbing its channel-based consensus.

#### 2.2.1 Rationale

- **A correction is a refinement, not a new observable.** C1 means "pixel A, but in this
  specific neighborhood configuration." The thing asserted about the world is still A's value;
  the refined identity lives in the neuron id and routing context, not in the coordinate. The
  coordinate genuinely does not change — inheriting it preserves a true invariant rather than
  inventing a convenient one.
- **It restores the temporal level-0 invariant.** The temporal machinery was built on the
  assumption that everything at `temporal_level_index[0]` is a coordinate-bearing token
  (channel / dimension / value). The apex handoff broke that by inserting coordinate-less
  pattern neurons. With inheritance, temporal level 0 is a uniform interface language again —
  some tokens raw, some context-refined — and the per-dimension consensus needs no changes.
- **Subsumption + inheritance compose.** When C1 fires, parent A is subsumed out of the apex,
  so the consensus sees exactly **one** vote-bearing token per recognized coordinate — no
  double-counting of parent and child on the same channel-dim.
- **Topology above L0 falls out:** inheriting the channel gives every level the parent's
  neighbor graph, and receptive fields still grow by composition — an L2 whose context is L1s
  anchored at A's neighbors effectively covers 5×5 even though its own neighbor set is A's 3×3.
- **Keeps the votes API uniform:** `FrameVote.value` is the dequantized bucket value, which is
  undefined for coordinate-less apex patterns. Inheritance makes the votes payload uniform across
  raw and refined voters.

#### 2.2.2 Implementation notes

- Inherit (channel, dimension, coordinate) at mint time in `allocate_spatial_pattern_neuron`
  (it already receives `parent_id`).
- **Do not register inherited coordinates in `neurons_by_value`** — that map requires
  coordinate uniqueness, and value→neuron resolution (action targets, event lookup) must
  always land on the L0 sensory/action neuron. Refined tokens are only ever reached via
  routing matches. The inherited coordinate is metadata for consensus grouping, neighbor
  filtering, and vote dequantization.
- Persistence: inherited coordinates need no new snapshot field — they are derivable on
  restore by walking `neuron_parents` to the L0 ancestor (parent ids are already persisted).
  Deeper levels chain: L2 inherits from its L1 parent, which anchors at the original L0
  coordinate.

#### 2.2.3 Identity matching in temporal processing — id-level, by design

An interaction to be aware of: the apex vocabulary shifts during early training. Temporal
patterns initially wire toward raw pixels; once spatial corrections start firing and subsuming
their parents, the apex contains C1 instead of A, and temporal predictions of A score as
misses under id-based error evaluation (`evaluate_vote_error` checks id membership in
actuals).

Error evaluation stays **id-level**: a refined token is not interchangeable with its anchor —
seeing a dog is not the same thing as seeing a brown pixel; the dog merely *uses the
coordinates* of that pixel for consensus bookkeeping. The model relies on **apex
stabilization**: once C1 exists and its context
recurs, the apex token for that configuration is C1, consistently — the system does not revert
to A. The transition period produces some temporal patterns wired to soon-obsolete tokens;
those stop activating and are reclaimed by the existing forgetting mechanism
(activation-strength decay / death ledger). This is accepted cost, not a problem to engineer
around. Note the explicit assumption this rests on: **without apex stabilization the design
does not work** — which is one more reason fixes 1.1 and 2.1 (which make recognition fire and
minting quench) are prerequisites, and one more diagnostic to watch (apex composition churn
over episodes should fall to ~zero on a fixed training set).

#### 2.2.4 Accepted consequence: consensus output is a lossy projection

With inheritance, the per-dimension inference output reports a refined token under its
anchor coordinate — temporal may predict "stroke fragment anchored at A" and the consensus
reports "px_A = on." Nothing is lost internally (routing and contexts use full neuron ids),
and the richer identity is recoverable from the votes payload. The consensus layer speaks the
observable vocabulary; the hierarchy speaks the structural one.

---

## 3. Expected dynamics after the fixes (notes from design discussion)

- **Apex wiring lag.** `learn()` wires `last_frame_apex`; corrections minted this frame only
  fire on the next exposure, so wiring chases the apex with a one-exposure lag, and each new
  hierarchy level temporarily disinherits the previous wiring. This is acceptable *provided*
  minting quenches — which fix 2.1 enables (modal predictions stop erroring once local stats
  stabilize). Expect per-episode train accuracy to climb and then stabilize; multi-episode
  training is required, not a bug.
- **Subsumption granularity** (design doc §8 item 3). With 1.1 + 2.1 + 2.2 fixed, every pixel
  whose neighborhood deviates from modal mints and later fires its own correction, so it is
  subsumed through its own parent slot — per-parent subsumption gives near-complete coverage
  without changing the rule. Residual to watch empirically: a pixel whose own neighborhood
  stayed modal (never erred, owns no correction) but which sits inside a neighbor's fired
  correction context stays apex alongside that correction — mild double-counting at pattern
  boundaries.
- **Accuracy expectations.** Post-fix, ~100% train accuracy on small sets is the expected
  outcome (first-learned patterns in default wiring + corrections for conflicting patterns, per
  design doc §4.6). Test accuracy remains bounded by exact-template Jaccard matching across
  instance variation.

---

## 4. Minor issues

- **`unwrap_or(0)` channel sentinel** ([thalamus.rs:1036, 1406, 1412](../brain/brain-core/src/thalamus.rs)
  and elsewhere): works only because channel ids start at 1 ([thalamus.rs:333](../brain/brain-core/src/thalamus.rs)).
  If a real channel id 0 ever exists, pattern neurons silently inherit its neighbor set.
  Propagate the `Option` instead. (Partially superseded by fix 2.2, which gives patterns a real
  inherited channel as part of the full coordinate.)
- **Committed platform binaries** — ⏳ OPEN (roadmap: *Do not commit platform binaries*): the branch
  updates `brain-napi.node` and adds a new ~1.5 MB `brain-napi.win32-x64-msvc.node`. Gitignore
  `*.node` rather than growing them in history.

---

## 5. Verified non-issues

Checked during review; listed to save future reviewers the time:

- The apex `fired_set` **does** include patterns activated mid-sweep — they are written into
  `spatial_level_index[level+1]` and picked up at the top of the next level iteration
  ([brain.rs:1156-1160](../brain/brain-core/src/brain.rs)); the sweep never breaks before
  processing a newly grown level.
- d=0 connection learning **is** neighbor-filtered per task ([thalamus.rs:1410-1417](../brain/brain-core/src/thalamus.rs)).
- Spatial state is ephemeral per frame (`reset_spatial`), `reset_context` clears
  `last_frame_apex`, and the learning flag is respected in all new paths (spatial error
  recording skipped in eval mode).
- Spatial order of operations is correct: votes are generated **before** d=0 connection
  learning, so votes reflect prior-frame state.
- Correction minting installs routing entries for *next* frame (matches design §4.4 "next
  time"); spatial corrections get `spatial_level = parent + 1`, temporal level 0.
- Encoder neighborhood math (radius loops, edge clipping, channel naming) is correct; the new
  MNIST jobs keep train/test cleanly separated.

---

## 6. Fix order

The three-fix dependency chain has **landed** in the order below; the two independents remain and
are scheduled on the [roadmap](roadmap.md).

**Dependency chain — ✅ all done, in this order:**

1. **1.1 — observed-context neighbor filtering. ✅** Until this lands, spatial matching never
   succeeds, so nothing downstream can be observed to work. Everything else is gated on it.
2. **2.1 — per-position winner competition. ✅** Turns the error signal into a Hamming distance from
   the modal patch (instead of novelty-vs-everything-seen), so minting quenches once local
   statistics stabilize. This is what produces the apex stabilization that 2.2 depends on.
3. **2.2 — coordinate inheritance for minted patterns. ✅** Restores locality above L0 and keeps
   temporal level 0 a uniform coordinate-bearing interface. Only meaningful once 1.1 and 2.1 have
   recognition firing and minting quenching.

The chain produced the observable changes predicted here (routing matches fire, hierarchy depth
grows past 1, minting quenches, train accuracy converges) and the 95.73% full-MNIST capstone.

**Independent — ⏳ still open, scheduled on the roadmap:**

- **1.3 — restore the temporal `< 1` guard.** One line; fixes a temporal-matching regression
  unrelated to the spatial chain. Roadmap: near-term fix.
- **1.2 — snapshot round-trip.** Required before any run that relies on persistence; includes the
  Phase 4 round-trip test. No dependency on the algorithm fixes. Roadmap: *Backups / imports /
  exports (Phase 4)*.

**Follow-on questions opened by the landed chain** (roadmap experiments, not bugs):

- **Temporal channel inheritance** — 2.2 gave *spatial* corrections their parent's coordinate;
  should *temporal* corrections inherit channels the same way? Tracked as its own roadmap item.
- **Context refinement** — re-introduce the consolidation step removed in `8a17f4d`, behind a flag,
  for both temporal and spatial. Roadmap item, synergistic with merge tuning.
