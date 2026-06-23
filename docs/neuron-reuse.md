# Neuron Reuse for Error Correction

This document is the **theory** of neuron reuse — motivation, mechanism, and how it interacts with
spatial processing. It mirrors [spatial-processing.md](./spatial-processing.md): it describes *what*
reuse is and *why* it works, not the build order. The implementation is split across four phase docs
plus a validation doc; see [§6 Implementation](#6-implementation) for the index.

---

## 1. Motivation

### 1.1 The Problem

Currently, when the thalamus detects an error, it always creates a brand new neuron. But the inference needed — the prediction, the grouping — may already exist somewhere in the network. A neuron created for a completely different context might already have connections that express exactly the required prediction.

Without reuse, the network grows indefinitely: every error event mints a fresh neuron, regardless of structural overlap with existing neurons.

### 1.2 The Solution: Reverse Inference Index Lookup + Batched Mint

Reuse applies to **all distances**, not just d=0. Any error (temporal or spatial) is a candidate for reuse before minting.

The reuse criterion is **inference-output match**, not context-match: does some existing neuron's connection set already produce the inference the correction would need? The candidate's own routing table and triggering context are irrelevant to the decision — only its output signature matters.

Correction operates **per (distance, neighborhood) per frame**, not per error. The frame's observed reality at a given distance *within a given receptive field* is singular (§2.1), so a single lookup resolves the correction target for every neuron that erred against that same local reality. On a lookup miss, a single correction neuron is minted for that reality and all co-failing neurons in that group are wired to it (the batched mint). Lookup handles cross-frame reuse — the source of generalization — while the batched mint dedups within-frame redundancy on a miss.

### 1.3 Why Reuse Is Essential, Not an Optimization

Reuse is what makes generalization possible.

A single correction neuron created from one event with no reuse only memorizes that event. Decay alone has no statistical basis to identify which connections are incidental.

When the same correction neuron is reused across many distinct error events, each reuse strengthens the connections shared across all those events and adds new connections specific to each event. The structurally-shared connections accumulate strength; the per-event-specific connections remain weak. Over many reuses, the neuron's strong connections converge on the structural core common to the equivalence class of triggering events.

Decay then sharpens this: incidental per-event connections erode while reinforced structural ones persist.

Reuse provides the cross-instance signal; decay sharpens it. Both are required.

---

## 2. Mechanism

### 2.1 One Observed Reality Per (Distance, Neighborhood)

The frame's reality is singular *within a receptive field*. At a given distance d, inside a given neuron's neighborhood, in a given frame, there is exactly **one** observed inference set — the actual co-activations (d=0) or the actual realized next state (d>0), filtered to that neighborhood. This is what happened there; it does not vary across the neurons that erred against it.

> **Correction to the earlier single-level model.** An earlier draft asserted "one observed reality per *distance*." That is too strong. Spatial error evaluation is **neighbor-filtered** — a neuron's observed co-activation set is its local neighborhood's fired set, not the whole frame ([thalamus.rs:1137-1144](../brain/brain-core/src/thalamus.rs), the "observed set (L0 events, neighbor-filtered)"). Two neurons in disjoint neighborhoods erring at the same distance are correcting toward *different* local realities and must not be merged. The grouping key is therefore **(distance, observed-set / neighborhood)**, and there may be more than one correction minted per distance per frame — one per distinct local reality. Everything below that says "per distance" should be read as "per (distance, neighborhood) group."

Multiple neurons may err against the same local reality, but they all erred by predicting *different wrong things* about the *same* observed set. Two sets must not be conflated:

- The **predicted** set — per-neuron, varies. This is *why* each neuron erred differently.
- The **observed** set — shared across all neurons erring against the same local reality. This is the single reality being corrected toward.

Correction always targets the observed set. Because the observed set is one thing per (distance, neighborhood) per frame, there is exactly one reality to correct toward per group — and therefore at most one correction neuron to mint per group. This singularity drives both the lookup (§2.2) and the batched mint (§2.3).

### 2.2 Per-Group Lookup

For each (distance, neighborhood) group in which any error occurred this frame:

1. The thalamus knows the **observed inference set** for that group — the one local reality from §2.1.
2. Query the **reverse inference index**: for each observed target T, which existing neurons have a connection to T at this distance? (This is the inverse of "neuron N infers targets {T1, T2, …}.")
3. Take the union of those candidate sets. For each candidate, score its inference signature against the observed set using the same common/missing/novel analysis as pattern recognition.
4. If a candidate scores above the **merge threshold for this distance** (`spatial_merge_threshold` at d=0, the temporal merge threshold at d>0 — see §2.5): wire **all** neurons erring in this group to defer to that candidate.
5. If no candidate qualifies: fall through to the batched mint (§2.3).

Note this is **one lookup per group per frame**, not one per error. Since the observed reality is shared within the group, a single query resolves the correction target for every neuron that erred against it — eliminating the per-error lookup fan-out.

### 2.3 Batched Mint (Fallback)

When the lookup misses — no existing neuron expresses the observed reality well enough — mint **one** correction neuron for that group's observed reality and wire **all** co-failing neurons in the group to it. Do not mint per erroring neuron.

This collapses the within-frame redundancy the lookup cannot catch: several neurons failing simultaneously against an identical not-yet-existing reality would otherwise each mint their own duplicate. The grouping key is the (distance, observed-set) pair (§2.1); within a group the observed reality is unambiguous and singular.

> This is a change from today's mint path. Currently spatial mints **one correction per erroring parent** ([thalamus.rs:1205-1229](../brain/brain-core/src/thalamus.rs)) and temporal mints **one per (neuron, age)** ([thalamus.rs:1471-1497](../brain/brain-core/src/thalamus.rs)). Batched mint replaces both with one-per-group. See [neuron-reuse-frame.md](./neuron-reuse-frame.md).

Per group, per frame, the correction path is therefore:

1. One lookup against the group's observed reality.
2. On hit: wire all co-failers to the existing neuron.
3. On miss: one batched mint, wire all co-failers to it.

### 2.4 Why Lookup Is Still Required

The batched mint is a *within-frame* collapse only. It does nothing across frames. The same reality recurring in a later frame is, to the mint path, a fresh frame with its own batched mint that knows nothing about the earlier one — so without lookup, every recurrence of a reality mints another duplicate.

Batching alone yields neuron count proportional to (distinct realities × recurrences). Lookup is what drives it down to (distinct realities).

More importantly, the cross-frame collapse *is* the generalization mechanism (§1.3): a neuron reused across many distinct error events accumulates the structural core while incidental connections decay. A neuron minted once and never reused across frames only memorizes that frame. Batching cannot produce this statistical signal because it never links one frame to another — and the transfer-learning effect (train on digits 0–4, reuse on 5–9; see [neuron-reuse-validation.md](./neuron-reuse-validation.md)) is inherently cross-frame and therefore invisible to batching.

Lookup and batched mint compose cleanly: lookup first (existing neuron wins, cross-frame generalization), batched mint as the fallback beneath it (within-frame dedup on a true miss). This composition is exactly why the build order is **batched mint first, then the index, then the lookup on top** — see §6.

### 2.5 Symmetry with Pattern Recognition

The symmetry is intentional. Pattern recognition asks "does this observed context partially match a stored context?" Reuse asks "does this required inference partially match an existing neuron's inference?" Both are partial-set-overlap questions; they share the same threshold.

Reuse reads the **same merge threshold pattern recognition uses at that distance** — and that threshold is now split per phase. Spatial reuse (d=0) reads `spatial_merge_threshold`; temporal reuse (d>0) reads the temporal merge threshold ([neuron.rs:200-205, 240](../brain/brain-core/src/neuron.rs)). There is no separate `reuseMergeThreshold` parameter on either side — the coupling to pattern recognition is intentional. Setting a phase's merge threshold to 1.0 disables reuse *and* partial-context recognition for that phase together.

### 2.6 Worked Example

Observed inference set in some neighborhood = (A, B, C). Candidate neuron infers (B, C). Overlap 2/3 ≈ 0.67. If the merge threshold for this distance is below 0.67, reuse. Every neuron that erred against this local reality this frame has its routing entry pointed to the candidate; when the same context recurs, the candidate fires and provides the (B, C) inference (missing A is accepted as the cost of reuse). Had no candidate qualified, a single new neuron would have been minted for (A, B, C) and all of this group's co-failers wired to it.

---

## 3. Interaction with Spatial Processing

### 3.1 Levels as Activation State, Not Neuron State

Reuse requires that a neuron's level be a property of its **activation this frame**, not a property stored on the neuron, because a single neuron R may be reused from different routing sources at different levels in different frames.

Much of this is **already true in the current code**. Active memory already carries per-activation level indexes — `spatial_level_index: Level → {NeuronId}` and `temporal_level_index: Level → FrameNumber → {NeuronId}` ([memory.rs:57,63](../brain/brain-core/src/memory.rs)) — and both sweeps already iterate them. What remains is a *separate persistent copy* of spatial level that reuse must remove: the thalamus-side map `neuron_spatial_levels: NeuronId → Level` ([thalamus.rs:219](../brain/brain-core/src/thalamus.rs)), read by mint (child = parent + 1), diagnostics, the sweep bound, and serialization. That intrinsic copy is what breaks under cross-level reuse.

Under the activation-state model: a neuron activated this frame is registered in active memory at `activating_neuron.activation_level + 1` (sensory neurons start at activation level 0). A correction minted from a parent activated at spatial level 2 takes level 3 *as an activation fact*, and is **not** activated this frame anyway (corrections match next frame via routing — [thalamus.rs:1262-1296](../brain/brain-core/src/thalamus.rs)); next frame, whichever source fires it sets its activation level afresh. R's identity is preserved; its "level" is per-frame contextual.

Without this, cross-level reuse would either be unsafe (the sweep might never reach R's intrinsic level, dropping R's d>0 work and votes) or require restricting reuse candidates to matching levels (shrinking the pool). With per-activation levels, neither problem exists.

The mechanics of removing the persistent map — and the eight readers that must move to activation-derived or recomputed values — are the entirety of [neuron-reuse-levels.md](./neuron-reuse-levels.md) (Phase A).

### 3.2 Refractory and Correction-Wired Inhibition

Reuse introduces two per-frame tracking sets that don't exist today:

**`fired_this_frame: FxHashSet<NeuronId>`** — every neuron that has fired in either phase this frame. Enforces refractory: each neuron fires at most once per frame. Mint-only correction doesn't need this because a freshly minted neuron is unique and there are no routing cycles into it; reuse can wire an existing neuron from many sources in the same frame, so refractory becomes load-bearing for termination.

**`correction_wired_this_frame: FxHashSet<NeuronId>`** — every neuron whose activation this frame is the result of being selected as a correction target, whether minted fresh or reused. Neurons in this set:

- **Learn from the current observed set.** Their d=0 connections strengthen toward the observed reality — this is how a reused neuron gradually generalizes across reuse events.
- **Do not vote this frame.** Their activation is a wiring side-effect, not an inferential signal. This exclusion is **layered on top of** the existing voting suppression (the `activated_pattern_id` per-age suppression at [memory.rs:197-206](../brain/brain-core/src/memory.rs)), not a replacement for it. When `correction_wired_this_frame` is empty — i.e., before reuse lands — voting is bit-identical to today.
- **Are not error-checked this frame.** Prevents a reused neuron whose pre-existing d=0 set doesn't match the current observed set from generating a fresh error and cascading into more corrections within the same phase.

This inhibition is the load-bearing termination rule. Without it, cross-frame reuse would risk runaway error cascades within a single spatial phase. Both tracking sets, and the corrected voting framing, land with the lookup in [neuron-reuse-final.md](./neuron-reuse-final.md) (Phase D) — they are dead weight before there is reuse to make them load-bearing.

Both sets clear at frame end.

### 3.3 Reuse Across Both Phases

Reuse applies in both `process_spatial` (d=0 errors) and `process_temporal` (d>0 errors). Same mechanism, different phase, different merge threshold (§2.5). The reverse inference index and the two tracking sets are shared between phases.

---

## 4. Benefits

- **Neuron count reduction**: No redundant neurons computing the same inference. The network stays compact.
- **Transfer learning, at matched receptive fields**: If two different contexts reuse the same neuron, they are inherently linked. Knowledge transfers structurally rather than through an explicit transfer mechanism. Note the scope limit: because spatial corrections inherit the parent's coordinate ([spatial-processing.md §4.4](./spatial-processing.md)), reuse fires for shared structure at the **same receptive-field location** — it is co-located transfer, not translation-invariant transfer. For centered domains (MNIST) this is the common case; the validation doc frames its transfer test accordingly.
- **Robustness**: Shared representations are stronger — reinforced from multiple activation pathways.
- **Convergence speed**: The system builds on existing structure rather than rebuilding from scratch in each context.
- **Content-addressable network**: The thalamus can answer "is there a neuron that does X?" efficiently via the reverse inference index. Structurally similar to the existing `spatial_context_index` / `temporal_context_index`, but indexing **connections** (target → sources) rather than **context** (ctx → patterns) — a genuinely new index, see [neuron-reuse-index.md](./neuron-reuse-index.md).

---

## 5. Risk Assessment

### 5.1 High Confidence

Reuse is correct in principle. The reverse-inference-index lookup is a straightforward extension of existing index machinery. The `correction_wired_this_frame` inhibition rule (§3.2) makes reuse safe under within-frame error cascades.

### 5.2 Key Risks

- **Reverse-index cost**: per-frame reuse lookup could dominate runtime if poorly indexed. Mitigation: one lookup per (distance, neighborhood) group per phase; shard by column for parallel evaluation across regions; apply index updates at orchestration boundaries so lookups always see fresh data.
- **Dead-edge pollution**: with no connection-delete path today (connections persist; decay was removed), a membership-only reverse index never sees removals. Currently harmless (every indexed edge is live, strength ≥ 1). If decay/delete is reintroduced, the index must drop edges on delete. Tracked in [neuron-reuse-index.md](./neuron-reuse-index.md).
- **Over-aggressive reuse**: too-low merge threshold causes inappropriate reuse, polluting reused neurons with mismatched contexts. Mitigation: the per-phase merge threshold means tuning is coupled to pattern recognition; default position is "tune cautiously and rely on decay to clean up bad reuses."
- **Action binding dilution**: heavily-reused correction neurons may have action votes diluted across many contexts. Mitigation: the long-run validation monitors this; normalization can be added if observed.

### 5.3 Two Decisions to Settle Before Building Phase C/D

Two semantic questions are not yet resolved and are called out as **DECIDE-THIS** blocks in the phase docs where they bite. They are surfaced here so they aren't lost:

1. **Mint-frame vs reuse-frame inhibition window** — when exactly does the "learn-but-don't-vote-don't-error-check" window apply: the frame a correction is minted/reused, or the first frame it re-fires via routing? See [neuron-reuse-frame.md](./neuron-reuse-frame.md) DECIDE-THIS #1.
2. **Refractory vs cross-level injection** — when a reused neuron has already fired this frame via its own routing match *and* is then selected as a correction target at a higher level, which wins, and at what level does it land for apex/temporal handoff? See [neuron-reuse-final.md](./neuron-reuse-final.md) DECIDE-THIS #2.

---

## 6. Implementation

The build order is **levels → index → batched mint → lookup**, then validation. Each engineering phase
lands behind its own gate so a regression has a single suspect. Batched mint precedes the index
deliberately: it is the correction-path reshape the lookup sits on top of, and it delivers a measurable
within-frame neuron-count win on its own (§2.4).

| Phase | Doc | Goal | Gate |
|---|---|---|---|
| **A** | [neuron-reuse-levels.md](./neuron-reuse-levels.md) | Remove the persistent `neuron_spatial_levels` map; derive spatial level from activation. Isolated bit-exact refactor + backup format bump. | Stocks regression **bit-exact**; backups round-trip; diagnostics unchanged. |
| **B** | [neuron-reuse-index.md](./neuron-reuse-index.md) | Build the reverse **inference** index (target → distance → sources) over both connection stores. Built and unit-tested, **not yet consumed**. | Unit: target→sources lookup correct across both stores; index size ∝ connection count; still bit-exact. |
| **C** | [neuron-reuse-frame.md](./neuron-reuse-frame.md) | Reshape the correction path to **batched mint**: group errors by (distance, neighborhood), mint one, wire all co-failers. First behavior change. Settles DECIDE-THIS #1. | MNIST neuron count drops from within-frame dedup alone; accuracy ≥ mint-only baseline. |
| **D** | [neuron-reuse-final.md](./neuron-reuse-final.md) | Reuse **lookup** on top of batched mint, plus the `fired_this_frame` / `correction_wired_this_frame` tracking sets and corrected voting. Settles DECIDE-THIS #2. | Unit: cross-frame reuse via lookup; MNIST neuron count drops further (target ≥30% vs A-baseline); reuse lookup < 20% per-frame overhead. |
| **Validation** | [neuron-reuse-validation.md](./neuron-reuse-validation.md) | Cross-domain experiments runnable only after D: MNIST transfer (0–4 → 5–9), stocks full-pipeline integration, forget-rate / class-neuron long-run. | Per-experiment gates in the doc. |

All phases depend on [spatial-processing](./spatial-processing.md) being complete (it is).
