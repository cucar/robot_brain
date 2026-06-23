# Neuron Reuse — Phase C: Batched Mint (Within-Frame Dedup)

**First behavior change, and the heaviest engineering phase.** Theory in [neuron-reuse.md §2.1–2.4, §3.4](./neuron-reuse.md).
This phase reshapes the correction path so that co-failing neurons are **clustered** and **at most one**
correction is minted per cluster, with **all** co-failers wired to it. Because wiring many co-failers to one
neuron makes that neuron **multi-parent at birth**, the multi-parent ownership/lifecycle/activation
machinery lands here too (it is inseparable from batched mint — see "Multi-parent" below). It does **not**
touch the reverse index or cross-frame reuse — those come in [Phase D](./neuron-reuse-final.md). It is
sequenced before the index deliberately: the lookup in Phase D slots in *on top of* this reshaped path
(lookup first, batched mint as fallback), and batching alone is a measurable within-frame neuron-count win
([neuron-reuse.md §2.4](./neuron-reuse.md)).

Two design decisions are settled in this doc before any code: **DECIDE-THIS #0** (how to cluster co-failers
and what anchor the shared correction gets) and **DECIDE-THIS #1** (mint-frame inhibition window).

---

## Goal

Replace per-erroring-neuron minting with **per-cluster** minting:

- **Spatial** today: one correction per erroring parent ([thalamus.rs:1205-1229](../brain/brain-core/src/thalamus.rs)).
- **Temporal** today: one correction per (neuron, age) ([thalamus.rs:1471-1497](../brain/brain-core/src/thalamus.rs)).

After this phase, both collect their errors, **cluster** the co-failers (DECIDE-THIS #0), and mint **one**
correction per cluster, wiring every co-failer in the cluster to it — plus the multi-parent machinery that
shared correction now requires.

---

## DECIDE-THIS #0 — How to cluster co-failers (and what anchor the shared correction gets)

This is the **central unresolved design question of Phase C**, and it must be settled before any batched
mint is built. An earlier draft keyed the batch by `(distance, observed_set)` and waved off the
alternatives as "either key works." That was wrong, for two reasons that turn out to be the same problem.

**Naive group-by doesn't cluster.** Spatial error evaluation is neighbor-filtered: each erroring neuron's
observed co-activation set is *its own* neighborhood's fired set ([thalamus.rs:1137-1144](../brain/brain-core/src/thalamus.rs)).
Every neuron has its *own* neighborhood center (one neuron per coordinate per level), so keying by
neighborhood-center yields groups of size 1 — **no batching at all**. Batching only happens if we group
neurons that observed the **same fired-set** (by neuron id) — and those neurons sit at **different
(near-neighbor) coordinates**. So batching across coordinates is inherent to batching doing anything.

**Which re-opens the anchor question.** A spatial correction inherits its parent's anchor coordinate
(§4.4) — written for **one** parent. A batch's co-failers are anchored at *different* coordinates, so the
shared correction's anchor is ambiguous. Connections crossing channels is normal and fine; the **anchor**
is the problem. Aggregating anchors (a multi-channel/dimension child) breaks: the child's **neighborhood**
becomes undefined (no single neighbor set for its own d=0 learning / error eval), **receptive-field growth**
(§4.4, one radius hop per level) loses its anchor, the **temporal uniform interface** (§3.3, every apex
token is one (channel, dimension, value)) loses its vocabulary, and **coordinate/label resolution** loses
its root. Aggregation is therefore out.

So the anchor must be a **representative** of the batch's parents — which is only sound if the batch's
parents are **close**. That makes batching a **neighborhood-aware clustering**, not a hash group-by. And
because neighborhoods overlap (a neuron belongs to many), there is no clean partition.

**Options (DECIDE):**

1. **Greedy neighborhood clustering.** Seed on an unclustered correction request; batch all
   same-observed-reality requests whose anchors fall within the seed's radius; representative anchor = seed
   (or centroid); remove and repeat. Deterministic with sorted seeds; overlap resolved first-come.
2. **Connected-components.** Edge between two requests if they share the observed reality *and* their anchors
   are within radius; batch = connected component; representative = min-id / centroid.
3. **Drop batched mint; defer all dedup to cross-frame reuse (Phase D).** If position-anchored corrections
   are deemed legitimately *distinct* within a frame (never真 duplicates), there is no within-frame
   redundancy to collapse — each erroring neuron mints its own, and the lookup handles dedup across frames.
   Simplest; loses within-frame collapse, but only if that collapse was conflating distinct anchors anyway.

Same-channel value blending is the easy sub-case (one channel, average the value into the representative);
cross-channel is the hard case the clustering must keep out of a single batch, or accept the representative
smear. **Until this is decided, the dispatch below is written generically as "group" — substitute the
chosen clustering.** This decision also fixes whether batched mint produces a **multi-level** correction:
clustering across spatial levels (co-failers at different levels) makes the shared R multi-level (needs
Phase A's multi-level memory); clustering within a level keeps R single-level.

---

## Design

### Spatial path

`mint_spatial_corrections` ([thalamus.rs:1090-1234](../brain/brain-core/src/thalamus.rs)) currently, per
fired parent: computes predicted vs observed (neighbor-filtered), and if error > threshold mints one
correction with context = the parent's neighborhood. Reshape to two passes:

1. **Collect**: for each erroring parent, emit `(erroring_parent_id, observed_set, anchor, spatial_level)`.
2. **Cluster** the collected requests per DECIDE-THIS #0 (greedy-neighborhood / connected-components / or
   none if option 3 is chosen). Each cluster carries a **representative anchor**.
3. **Mint once per cluster**: allocate **one** correction (context = the cluster's observed set, anchor =
   the representative), then wire **every** erroring parent in the cluster to it via
   `install_spatial_corrections` ([thalamus.rs:1262-1296](../brain/brain-core/src/thalamus.rs)) — each
   parent gets a routing entry pointing at the shared correction. The install path must now place **one
   child into many parents' routing tables** (see "Multi-parent" below) — today it installs per-parent.

The error-feedback accumulation (`error_feedback: Vec<(NeuronId, f64)>` at
[thalamus.rs:1102, 1203](../brain/brain-core/src/thalamus.rs)) is **per erroring neuron** and stays
per-neuron — grouping changes who-mints-what, not who-recorded-an-error. Each co-failer still records its
own error sample.

### Temporal path

Temporal mints inline, one per (neuron, age), inside the per-(neuron, age) evaluation
([thalamus.rs:1471-1497](../brain/brain-core/src/thalamus.rs), driven by `evaluate_vote_error` at
[thalamus.rs:1711-1745](../brain/brain-core/src/thalamus.rs)). Reshape to collect-then-mint:

1. **Collect**: evaluate every (neuron, age); for each error, emit
   `(group_key = (distance = age, observed_event_set), erroring_neuron_id, age)`. The observed set at d>0 is
   the realized next-state event set (neighbor-filtered, same as the predicted/observed comparison in
   `evaluate_vote_error`).
2. **Cluster + mint once per cluster** (per DECIDE-THIS #0, keyed on age + observed-event-set + anchor):
   one temporal pattern per cluster with pre-wired connections per the existing connection-spec builder
   ([thalamus.rs:431-487](../brain/brain-core/src/thalamus.rs)); wire all co-failing (neuron, age) pairs in
   the cluster to it.

> Note the distance/age relationship: temporal "distance" in the connection spec is `age - a`
> ([thalamus.rs:450](../brain/brain-core/src/thalamus.rs)). The cluster is keyed by the **age** at which the
> error occurred (the prediction horizon), which is the index Phase D's lookup will query. Keep the two
> notions distinct in code.

### Dispatch shape (per phase)

```
errors  = collect_errors_from_phase_stabilization()
clusters = cluster(errors)                      // DECIDE-THIS #0; representative anchor per cluster
for c in clusters:
    n = mint_one(c.observed, c.distance, c.anchor)   // ONE mint for the whole cluster
    for e in c.errs: wire_correction(e.erroring_neuron, n)   // n now has many parents — see Multi-parent
```

In Phase D this becomes `lookup(c.observed, c.distance)` first, `mint_one` only on miss — the loop body is
the seam.

---

## DECIDE-THIS #1 — Mint-frame vs reuse-frame inhibition window

The correction-wired inhibition ([neuron-reuse.md §3.2](./neuron-reuse.md)) says a corrected neuron
**learns the observed set but does not vote and is not error-checked** during its inhibition window. *Which
frame is that window?*

**The tension.** Today, spatial corrections are **not activated the frame they are minted** — they are
installed as routing entries and match on a **later** frame ([thalamus.rs:1262-1296](../brain/brain-core/src/thalamus.rs):
"patterns NOT activated this frame, match next frame"). But `correction_wired_this_frame`'s "learn from the
current observed set" only makes sense if the correction is *active now* to learn from it. And the code does
emit `correction_activations` out of the spatial pass ([neuron.rs:1101](../brain/brain-core/src/neuron.rs)),
suggesting corrections *do* produce activations the mint frame. These two readings must be reconciled before
the tracking sets in Phase D are meaningful.

**Recommended resolution (confirm against code during this phase):**

- A freshly minted correction **does** activate its mint frame, as a tagged `correction_activation`: it is
  added to `correction_wired_this_frame`, learns the group's observed set, **does not vote**, **is not
  error-checked**. This is its one-frame inhibition window.
- On **subsequent** frames it fires via a routing match like any recognized pattern: it votes and is
  error-checked normally. It is *not* in `correction_wired_this_frame` on those frames.
- The "match next frame" comment then refers specifically to the **routing-table** match path (recognition
  on recurrence), not to whether the neuron activates at all on the mint frame.

**Action this phase:** trace `correction_activations` from [neuron.rs:1101](../brain/brain-core/src/neuron.rs)
through the thalamus to confirm whether minted corrections are inserted into `spatial_level_index` the mint
frame. Write the answer into [neuron-reuse.md §3.2](./neuron-reuse.md) as settled fact. Phase D's tracking
sets are built against whatever this resolves to.

> Batched mint alone (this phase, no reuse) mints **fresh** neurons, which already carry the existing
> fresh-mint exemption (newly-minted error patterns skip the learn/vote/error-check steps —
> [spatial-processing.md §3.3 step 2](./spatial-processing.md)). So Phase C needs **no** new
> `correction_wired_this_frame` set: the exemption it would provide is already provided for fresh mints.
> That set becomes load-bearing only in Phase D, where a *reused pre-existing* neuron needs the same
> exemption it doesn't get for free. Settling DECIDE-THIS #1 here, though, is what lets Phase D add it
> cleanly.

---

## Multi-parent: batched mint is the first producer

Theory in [neuron-reuse.md §3.4](./neuron-reuse.md). **Batched mint makes a neuron multi-parent the moment
it is minted** — wiring k co-failers to one correction gives that correction k parents, in one frame, with
no lookup involved. So the multi-parent ownership/lifecycle/activation machinery lands **here**, not in
Phase D. (Phase D's lookup only adds a *second* parent to a neuron that already existed *before* this frame
— the same machinery, applied across frames.) Today a correction is owned by one host: its strength lives
in that host's routing entry and it dies when that one entry decays to zero
([neuron.rs:64, 689-703, 709-714](../brain/brain-core/src/neuron.rs)). The forget rate is **not** per-host —
`pattern_forget_rate` is brain-wide and uniform across every pattern neuron at every level
([brain.rs:366](../brain/brain-core/src/brain.rs), [thalamus.rs:492](../brain/brain-core/src/thalamus.rs)) —
so the per-parent entries all decay at the **same** rate, differing only in `activation_strength` and
`last_activation_frame`.

The work this phase must land:

- **Install one child into many parents' routing tables.** `install_spatial_corrections`
  ([thalamus.rs:1262-1296](../brain/brain-core/src/thalamus.rs)) and the temporal install path currently wire
  one correction into one parent; batched mint installs one correction into every co-failer's table. The
  routing-table structures (keyed child → entry, [neuron.rs:222-223, 242-243](../brain/brain-core/src/neuron.rs))
  already permit the same child id across many parents' maps — the work is in the install/lifecycle, not the
  storage.
- **Thalamus activation of a shared neuron.** Starting the frame *after* a batched mint, more than one of a
  correction's parents can match-and-activate it in the same sweep. The thalamus must collapse this to one
  activation (refractory, via `fired_this_frame`) while crediting **every** activating parent: each routing
  entry strengthened, and each marked subsumed. Today an activation carries one `activation.parent_id` and
  marks exactly that parent subsumed ([brain.rs:1151-1157, 1255-1261](../brain/brain-core/src/brain.rs)); with
  N activating parents, **all N** must be subsumed. → `fired_this_frame` lands here (it was wrongly scoped to
  Phase D).
- **Refcounted reaping.** A correction must be reaped only when **no** parent's routing entry references it —
  alive-if-referenced, not dead-when-one-host-decays. The cascade-delete path
  (`DeleteNeuron { target_id, parent_id }`, [column.rs:47, 218, 258-322](../brain/brain-core/src/column.rs),
  reaped around [thalamus.rs:1970](../brain/brain-core/src/thalamus.rs)) carries a **single** `parent_id`;
  replace it with a refcount over all referencing parents and scrub **every** parent's routing entry on
  death. Get this wrong and batched corrections leak (never reaped) or vanish while still referenced.
- **Multi-parent serialization.** `patterns.csv` = `pattern,parent,strength`
  ([backup.rs:13, 204-232, 412-436](../brain/brain-core/src/backup.rs)) serializes a pattern under **one**
  parent. Batched corrections need **many** `(parent, strength)` rows per pattern. Coordinate with Phase A's
  backup format bump — widen the schema once.
- **`parent_id` reader audit.** See the table below; every site assuming one parent.

### `parent_id` reader audit

Bucketed by whether multi-parent breaks it. The pattern: the **reverse** direction (who references me as
context) is already a set; the **forward** direction (a pattern → its owning parent) is the single-valued
side that breaks. Fix forward; leave reverse alone.

| Site | Current meaning | Multi-parent treatment |
|---|---|---|
| `patterns.csv` = `pattern,parent,strength` + `neuron_parents: pattern → parent` ([backup.rs:13, 204-232, 412-436](../brain/brain-core/src/backup.rs)) | pattern serialized as child of **one** parent | **Breaks.** Serialize pattern → **many** `(parent, strength)` rows; restore rebuilds an entry in every parent's table. Coordinate with Phase A's format bump. |
| `get_neuron_parent` ([brain.rs:785, 2053](../brain/brain-core/src/brain.rs)) → one parent | the unique parent | Return the **set**, or a designated canonical parent (the representative anchor) for chain-walks. |
| Coordinate / label resolution walks the parent **chain** to root sensory ([brain.rs:2048-2054](../brain/brain-core/src/brain.rs)) | linear chain | **Becomes a DAG.** Use the **representative anchor** from DECIDE-THIS #0 as the canonical parent for the walk. |
| `InspectedNeuron.parent_id: Option<NeuronId>` ([brain.rs:137, 789](../brain/brain-core/src/brain.rs)) | one parent + its context | Set of parents. Diagnostic only; low risk. |
| Subsumption marks `activation.parent_id` ([brain.rs:1151-1157, 1255-1261](../brain/brain-core/src/brain.rs)) | per-activation source | Per-activation source is fine, **but** a shared neuron fired from N parents ⇒ N subsumptions (see "Thalamus activation"). |
| `DeleteNeuron { target_id, parent_id }` + cascade ([column.rs:47, 218, 258-322](../brain/brain-core/src/column.rs)) | reap + tell **the** parent to unwire | **Breaks.** Reap over refcount; scrub **all** referencing parents' routing entries. |
| Context-ref scrub: `RemoveTemporal/SpatialContextRef`, `get_*_context_refs` ([column.rs:56-58, 264-275](../brain/brain-core/src/column.rs); [neuron.rs:449-458](../brain/brain-core/src/neuron.rs)) | already iterates **all** referencing parents | **No change** — reverse-ref side is already a set. |
| `ProcessResult.parent_id` ([column.rs:21-25, 163, 194](../brain/brain-core/src/column.rs)) | tag = the neuron that produced the task result | **No change** — labels the producing neuron, not a child's parent set. |

> Death-frame / persistence: project convention recomputes death frames from materialized strengths on
> restore. Under multi-parent that recompute is per-(parent, child) entry — unchanged in shape, no longer
> unique per child. Compute the reap decision over the **union** (alive if any entry alive).

---

## Acceptance gates (inline)

*(Gates assume DECIDE-THIS #0 resolved to a clustering option; if option 3 — drop batched mint — the batch
gates collapse to "each erroring neuron mints its own" and only the MNIST gate applies.)*

- **Unit — cluster batch**: co-failers the chosen clustering puts in one cluster produce **exactly one**
  minted correction, all wired to it (k routing entries → 1 child).
- **Unit — clustering boundary**: co-failers the clustering puts in *different* clusters produce separate
  corrections (no over-merge). Guards the clustering policy.
- **Unit — temporal batch**: two (neuron, age) errors clustered together at the same age produce one
  correction, both wired.
- **Unit — shared activation, all parents subsumed**: the frame after a batched mint, two parents both
  match-and-activate the shared correction. It fires **once** (refractory), and **both** parents are marked
  subsumed; each parent's routing entry is strengthened.
- **Unit — refcounted reaping**: a batched correction with two parents survives one parent's entry decaying
  to death (still referenced); is reaped only when the second dies. Confirm it is **not** reaped on the
  first host's death (the single-host regression this kills).
- **Unit — multi-parent serialization round-trip**: a batched correction with two parents snapshots and
  restores with both parents' routing entries intact (independent strengths), then continues identically.
- **MNIST**: rerun the MNIST harness. **Total neuron count drops** vs the Phase-A baseline (within-frame
  dedup). **Accuracy ≥ the Phase-A baseline** — batching must not cost accuracy.
- **Per-neuron error feedback unchanged**: each co-failer still records its own error sample; the Welford
  error stats over a fixed run match per-neuron expectations (clustering changed minting, not error
  accounting).

---

## Notes / gotchas

- **Wiring fan-out**: a cluster with k co-failers produces k routing entries (one per parent) to **one**
  correction. The install path must take a list of parents for a single child — see "Multi-parent."
- **Anchor of the shared correction**: the **representative anchor** from DECIDE-THIS #0, not an aggregate.
  Determinism: tie-break to the lowest-id parent, consistent with the existing "prefer smaller id"
  ([neuron.rs:1146](../brain/brain-core/src/neuron.rs)).
- **Determinism**: clustering and cluster-iteration order must be deterministic for comparable runs —
  sorted seeds / sorted keys, not hash order.
