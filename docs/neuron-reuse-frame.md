# Neuron Reuse — Phase C: Batched Mint (Within-Frame Dedup)

**First behavior change in the reuse plan.** Theory in [neuron-reuse.md §2.1–2.4](./neuron-reuse.md). This
phase reshapes the correction path so that, per (distance, neighborhood) group, **at most one** correction
neuron is minted and **all** co-failing neurons in the group are wired to it. It does **not** touch the
reverse index or cross-frame reuse — those come in [Phase D](./neuron-reuse-final.md). It is sequenced
before the index deliberately: the lookup in Phase D slots in *on top of* this reshaped path (lookup first,
batched mint as fallback), and batching alone is a measurable within-frame neuron-count win
([neuron-reuse.md §2.4](./neuron-reuse.md)).

---

## Goal

Replace per-erroring-neuron minting with **per-group** minting:

- **Spatial** today: one correction per erroring parent ([thalamus.rs:1205-1229](../brain/brain-core/src/thalamus.rs)).
- **Temporal** today: one correction per (neuron, age) ([thalamus.rs:1471-1497](../brain/brain-core/src/thalamus.rs)).

After this phase, both collect their errors, group by **(distance, observed-set / neighborhood)**, and mint
**one** correction per group, wiring every co-failing neuron in the group to it.

---

## Why the grouping key is (distance, neighborhood), not distance

Spatial error evaluation is **neighbor-filtered**: a neuron's observed co-activation set is its local
neighborhood's fired set, not the whole frame ([thalamus.rs:1137-1144](../brain/brain-core/src/thalamus.rs),
the "observed set (L0 events, neighbor-filtered)"). Two neurons in disjoint neighborhoods that both err at
d=0 are correcting toward **different** local realities and must not be merged into one correction. The
group key is the pair `(distance, observed_set)` — equivalently `(distance, neighborhood/coordinate)`,
since the observed set is determined by the neighborhood. There may therefore be **several** corrections
minted per distance per frame, one per distinct local reality — but at most one per group.

The observed set within a group is singular (that is the whole §2.1 argument), so there is **no
equivalence-class clustering to perform inside a group** — the grouping key already captures it.
Implementation: key the batch map by `(distance, hash(observed_set))`, or by `(distance, neighborhood_id)`
if the neighborhood identity is cheaper to obtain than hashing the set. Both erroring neurons in a group
share the same observed set by construction, so either key works; pick whichever the existing mint code
already has in hand at error-collection time.

---

## Design

### Spatial path

`mint_spatial_corrections` ([thalamus.rs:1090-1234](../brain/brain-core/src/thalamus.rs)) currently, per
fired parent: computes predicted vs observed (neighbor-filtered), and if error > threshold mints one
correction with context = the parent's neighborhood. Reshape to two passes:

1. **Collect**: for each erroring parent, emit `(group_key = (0, observed_set), erroring_parent_id, observed_set, context)`.
   Group by `group_key`.
2. **Mint once per group**: for each group, allocate **one** correction (context = the group's
   neighborhood/observed set), then wire **every** erroring parent in the group to it via
   `install_spatial_corrections` ([thalamus.rs:1262-1296](../brain/brain-core/src/thalamus.rs)) — each
   parent gets a routing entry pointing at the shared correction.

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
2. **Mint once per group**: one temporal pattern per group with pre-wired connections per the existing
   connection-spec builder ([thalamus.rs:431-487](../brain/brain-core/src/thalamus.rs)); wire all
   co-failing (neuron, age) pairs in the group to it.

> Note the distance/age relationship: temporal "distance" in the connection spec is `age - a`
> ([thalamus.rs:450](../brain/brain-core/src/thalamus.rs)). The group key's distance is the **age** at which
> the error occurred (the prediction horizon), which is the index Phase D's lookup will query. Keep the two
> notions distinct in code; the group key is keyed by age.

### Dispatch shape (per phase)

```
errors = collect_errors_from_phase_stabilization()
by_group = errors.group_by(|e| (e.distance, e.observed_set))   // observed reality is shared per group
for ((distance, observed), errs) in by_group:
    n = mint_one(observed, distance)          // ONE mint for the whole group
    for e in errs: wire_correction(e.erroring_neuron, n)
```

In Phase D this becomes `lookup(observed, distance)` first, `mint_one` only on miss — the loop body is the
seam.

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
> [spatial-processing.md §3.3 step 2](./spatial-processing.md)). So Phase C can ship **without** the
> `correction_wired_this_frame` set: the exemption it would provide is already provided for fresh mints.
> The set becomes load-bearing only in Phase D, where a *reused* (pre-existing) neuron needs the same
> exemption it doesn't get for free. Settling DECIDE-THIS #1 here, though, is what lets Phase D add it
> cleanly.

---

## Acceptance gates (inline)

- **Unit — spatial batch**: two parents erring against the **same** neighbor-filtered observed set at d=0
  in one frame produce **exactly one** minted correction, both wired to it.
- **Unit — distinct neighborhoods stay separate**: two parents erring at d=0 against **different**
  observed sets produce **two** corrections (no over-merge). This guards the (distance, neighborhood) key.
- **Unit — temporal batch**: two (neuron, age) errors against the same observed next-state set at the same
  age produce one correction, both wired.
- **MNIST**: rerun the MNIST harness. **Total neuron count drops** vs the Phase-A baseline (within-frame
  dedup alone). **Accuracy ≥ the Phase-A baseline** — batching must not cost accuracy.
- **Per-neuron error feedback unchanged**: each co-failer still records its own error sample; the Welford
  error stats over a fixed run match per-neuron expectations (grouping changed minting, not error
  accounting).

---

## Notes / gotchas

- **Wiring fan-out**: a group with k co-failers produces k routing entries (one per parent) to **one**
  correction — not k corrections. Confirm `install_spatial_corrections` / the temporal install path can
  take a list of parents for a single child without re-allocating per parent.
- **Context of the shared correction**: the correction's context is the group's observed set/neighborhood.
  If two parents in a group have *slightly* different neighborhoods but the same observed set (possible at
  boundaries), decide whose neighborhood/coordinate the shared correction inherits — recommend the
  lowest-id parent for determinism, consistent with the existing "prefer smaller id" tie-break
  ([neuron.rs:1146](../brain/brain-core/src/neuron.rs)). Strictly, if observed sets are equal the
  neighborhoods coincide by construction; this only bites if the group key is `neighborhood_id` rather than
  `hash(observed_set)`.
- **Determinism**: group iteration order must be deterministic for bit-comparable runs. Iterate groups in a
  sorted key order, not hash order.
