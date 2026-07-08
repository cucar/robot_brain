# Trial Counting and Economic Forgetting

Scope: two related fixes to the spatial (d=0) pattern lifecycle in
[`brain/brain-core/src/neuron.rs`](../brain/brain-core/src/neuron.rs) — consolidating a variable that was
wrongly split into two, and replacing flat time-based decay with a price-based earn/rent/death model. Both
build on the recognition and correction-creation machinery described in [algorithm.md](./algorithm.md); read
that first for `rank`, `price`, and `ALPHA`. Temporal (d>0) is out of scope for both — see
[Scope: spatial only](#scope-spatial-only-for-now).

## Part 1 — `fires` and `activation_strength`: one concept, wrongly split

### The bug

`SpatialRoutingEntry` currently tracks two separate numbers for what should be one idea — "how much has this
pattern actually fired":

- `fires: u64` — an immortal counter. Incremented by 1 on every accepted recognition win
  (`update_likelihood_model`), never decreases, never decays. Feeds `n_c` in `rank_by_likelihood_ratio`'s
  Bayesian estimate `p_c = (strength + ALPHA) / (n_c + 2)`.
- `activation_strength: f64` — a decaying eligibility gauge. Incremented by a flat `1.0` on every accepted
  fire (`strengthen_child_activation`, called immediately after `fires += 1`, same event), and eroded over
  elapsed frames by `pattern_forget_rate` when read (`get_child_effective_activation_strength`). Used only to
  decide whether a pattern is still a live candidate.

There is no biological basis for keeping a permanent, never-forgetting ledger of total activations separate
from the thing that decays — a pattern's current activation level already *is* the (time-discounted) memory
of its firing history. The split produces two concrete symptoms:

- **A birth anomaly.** Minting is two disconnected operations: `add_spatial_child` sets `fires: 0`, then a
  separate call to `strengthen_child_activation` bumps `activation_strength` to `1.0`. A freshly minted
  pattern is a live candidate (`activation_strength > 0`) with zero recorded trials (`fires = 0`) — the
  founding observation that created the pattern isn't counted as its first trial.
- **Confidence never fades.** Because `fires` never decays, a pattern that fired 50 times long ago and has
  gone dormant since still reports full statistical confidence (`n_c = 50`) in recognition, even though its
  `activation_strength` has decayed toward death. Liveness and confidence should erode together; today only
  one of them does.

### The fix

Delete the immortal counter. Recognition's trial count and the decaying activation gauge become the same
field, incremented by a flat unit per fire and decayed the same way `activation_strength` already decays:

```
n_c  =  get_child_effective_activation_strength(pattern_id, current_frame)   // decayed trial count
p_c  =  (strength + ALPHA) / (n_c + 2)                                       // unchanged formula
```

Birth stops being two operations: the founding observation is the pattern's first activation, so the unified
field starts at `1`, not `0` — `n_c = 1` at birth is the correct non-degenerate starting state, with no
separate "has this ever fired" bookkeeping required.

This field keeps its current name and role for the eligibility check (`get_child_effective_activation_strength
<= 0.0` gates candidacy); Part 2 below adds a second, separately-decaying field for economic survival — the two
are not merged into each other. See [Part 2](#part-2--price-based-forgetting-earn--rent--death) for why a flat
per-fire increment (this field) and a rank-weighted per-fire increment (the new balance) cannot share one
number.

### Sites to change

| File | What |
|---|---|
| `neuron.rs` — `SpatialRoutingEntry` | Remove the `fires: u64` field |
| `neuron.rs` — `add_spatial_child` | Remove `fires: 0` initializer |
| `neuron.rs` — `rank_by_likelihood_ratio` | Replace `n_c = entry.fires + 1` with the decayed activation read; thread `current_frame` through `rank_spatial_candidate` (its caller, `evaluate_spatial_candidate`, already has it) |
| `neuron.rs` — `update_likelihood_model` | Remove `entry.fires += 1` — the adjacent `strengthen_child_activation` call already performs the one real increment for this event |
| `neuron.rs` — `serialize_children` | Stop copying `entry.fires` into the snapshot; drop the temporal-branch placeholder `fires: 0` |
| `neuron.rs` — `SerializedChild` | Remove the `fires: u64` field |
| `column.rs` — `load_neuron` | Remove `entry.fires = child.fires` on restore |
| `backup.rs` — `patterns.csv` read/write | Drop the `fires` column entirely — `activation_strength` is already persisted as the `strength` column, so nothing new needs writing |

No NAPI surface exposes `fires` (`brain-napi` was checked) and no existing unit test asserts on it directly, so
this is contained to `brain-core`.

## Part 2 — Price-based forgetting (earn / rent / death)

### Why flat decay never worked

Every `SpatialRoutingEntry` decays under one global `pattern_forget_rate`, applied uniformly regardless of the
pattern's size or the value it delivers. This is why `--forget-rate 0` is the only mode that has ever produced
usable results: any nonzero rate kills patterns at a speed unrelated to whether they're worth keeping. Nothing
is ever deleted today — growth is append-only. Symptoms already measured:

- Later training episodes keep minting new patterns with diminishing held-out accuracy — a low-value episode
  can add tens of thousands of patterns for a fraction of a point of accuracy.
- An "apex tower" of neurons that fire once during minting and never again, permanently occupying the
  codebook.
- Obsolete scaffolding — ancestor patterns a more specific child has started subsuming — never yields, because
  nothing prunes patterns that have stopped being useful, only patterns that have stopped being *recent*.
- No convergence: codebook size only grows, so recognition candidate sets only grow, so recognition only gets
  slower.

### The three-part economic model

The death-cascade machinery (context scrubbing, routing-table removal) already exists and is reused unchanged.
What changes is the currency: instead of a flat, size-blind, score-blind strengthen/decay cycle, a pattern
carries a bit-denominated balance that it must earn back before its own price bankrupts it.

**Earning.** `rank_by_likelihood_ratio`'s score is already computed at every accepted fire, and is already
guaranteed positive (`approve_info_winner` requires `rank > 0.0` to fire at all) — it is the exact number of
bits by which this pattern explains the observation better than background statistics. That is the earning
event:

```
balance += rank        // per accepted fire
```

A crisp, frequently-firing pattern banks steadily; a pattern that never fires, or fires with only marginal
scores, banks nothing. (Deferred: a subsumption credit for a parent whose error machinery is silenced by a
child firing in its place — real value delivered, but detecting it requires additional plumbing in
`check_spatial_prediction`; not required to get the core mechanism running.)

**Rent.** Every error_info-minted pattern already carries `price: f64` — the exact
`name_bits(context) + name_bits(targets)` cost computed at birth ([algorithm.md](./algorithm.md#2-what-does-naming-this-shape-cost--the-price)).
Rent reuses that number directly instead of introducing a separate size term:

```
rent_per_frame = price / horizon
balance -= rent_per_frame     // accrued per frame (or per firing opportunity, see below)
```

A 40-entry context costs more per frame than an 8-entry one automatically, because its price is bigger — this
is the genuinely MDL part: big, blurry patterns must earn big to justify themselves, small crisp ones live
cheap.

**Death.** Unchanged mechanism, new inputs. `compute_death_frames`'s formula (frames until balance crosses
zero) stays structurally identical — fed `balance` and `price / horizon` instead of `activation_strength` and
`pattern_forget_rate`. Patterns die of unprofitability, not old age.

### Horizon

`horizon` is how many frames a pattern gets to earn back its price before rent alone bankrupts it — the
"expected payback period." A fixed frame count is rejected outright as an arbitrary constant. Instead:

```
horizon = current_frame / codebook_size
```

`codebook_size` is the same denominator already threaded through pricing (`neurons_by_value.len() +
paid_spatial_patterns.len()`). This is the average number of frames the system has taken, so far, to mint one
pattern — derived from the run's own history, not tuned. Early on, when the codebook is small and growing
fast, horizon is short: patterns must prove themselves quickly while vocabulary is still cheap to grow. Late
in training, when the codebook is large and growth has slowed, horizon lengthens: mature patterns get more
patience. Guard `codebook_size` at a floor (matching the existing `.max(2)` convention used elsewhere for
pricing denominators) so an early frame with a codebook of 1 doesn't produce a degenerate one-frame horizon.

Rent accrual per calendar frame is the first cut. Accruing rent only on frames where the pattern's parent was
actually active (a real "opportunity" to fire) is more correct — a pattern nobody has given a chance to prove
itself shouldn't bleed — but requires new per-child opportunity bookkeeping that doesn't exist today. Deferred
until the per-frame version is running and dormant-but-unproven patterns can be observed dying unfairly in
practice.

A bankrupt pattern's neuron may re-earn a slot later through the ordinary recurrence/minting gate — death is
cheap and reversible, which is what lets the reaper be aggressive.

### Why `balance` is not the same field as Part 1's trial count

Part 1 unifies `fires` and `activation_strength` into one flat, per-fire-incrementing, decaying trial count,
because both were tracking "how many times has this fired" and needed to agree. `balance` looks like the same
kind of quantity — it also decays, it also grows per fire — but it cannot be the same field:

- The trial count (`n_c`) must increment by a fixed unit per fire, because it stands in for "number of
  observations" in a Bayesian proportion estimate (`p_c = (strength + ALPHA) / (n_c + 2)`). If `strength`
  (an integer count of matches) is divided by a denominator that isn't itself counting observations,
  `p_c` stops being a probability.
- `balance` increments by `rank`, which is a variable, sometimes-large bit score, not a per-observation unit.
  A pattern that fires 3 times with strong matches (`rank ≈ 150` each) reaches `balance ≈ 450` — feeding that
  into the recognition formula as `n_c` would compute `p_c = (3 + 1) / (450 + 2) ≈ 0.009` for an entry that
  matched on every single fire, which is nonsense.

So two decaying fields, not one: a flat-increment trial count for recognition, and a rank-weighted,
price-rented balance for survival. Both decay — nothing left in the system is an immortal counter — they just
answer different questions and cannot substitute for each other.

### Scope: spatial only, for now

This applies to spatial patterns only. Temporal (d>0) children have no `price` or likelihood-ratio machinery
yet — that conversion is a separate, unstarted piece of work — so temporal keeps the current flat
`pattern_forget_rate` decay until it does.

## Open questions

- **Per-context-entry `strength` decay.** Part 1's trial count (`n_c`, the denominator) will decay, but the
  per-entry `strength` values that feed the numerator currently only ever increase (`+= 1.0` on a match, no
  erosion). If the denominator shrinks with dormancy but the numerator doesn't, an old, rarely-refreshed entry
  could report a distorted `p_c`. Whether `strength` needs its own decay in step with `n_c`, and by what rate,
  is unresolved — no formula for this has been derived yet, and one should not be picked arbitrarily.
- **Subsumption credit.** Deferred per [Earning](#the-three-part-economic-model) above — crediting a parent
  when a child fires and subsumes it is real value delivered, but not required for the core earn/rent/death
  loop to run.
- **Per-opportunity rent.** Deferred per [Rent](#the-three-part-economic-model) above — more correct than
  per-calendar-frame accrual, but needs new bookkeeping that doesn't exist yet.
