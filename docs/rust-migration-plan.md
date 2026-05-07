# Rust Core Migration Plan

## Goal

Migrate the brain's core computation from single-threaded JavaScript to a Rust core with:
- **MPI processes** → regions (inter-process communication)
- **Threads per MPI process** → columns (shared-memory parallelism)
- **Neurons** → owned by columns at various hierarchy levels

```
┌─────────────────────────────────────────────────────────────────────┐
│                        MPI Cluster                                  │
│                                                                     │
│  ┌─────────────────────┐    ┌─────────────────────┐                 │
│  │ Region 0            │    │ Region 1            │   ...           │
│  │ (MPI Rank 0)        │    │ (MPI Rank 1)        │                 │
│  │                     │    │                     │                 │
│  │  ┌──────┐ ┌──────┐  │    │  ┌──────┐ ┌──────┐  │                 │
│  │  │Column│ │Column│  │    │  │Column│ │Column│  │                 │
│  │  │  0   │ │  1   │  │    │  │  0   │ │  1   │  │                 │
│  │  │ N N  │ │ N N  │  │    │  │ N N  │ │ N N  │  │                 │
│  │  │ N N  │ │ N N  │  │    │  │ N N  │ │ N N  │  │                 │
│  │  └──────┘ └──────┘  │    │  └──────┘ └──────┘  │                 │
│  │  ┌──────┐ ┌──────┐  │    │  ┌──────┐ ┌──────┐  │                 │
│  │  │Column│ │Column│  │    │  │Column│ │Column│  │                 │
│  │  │  2   │ │  3   │  │    │  │  2   │ │  3   │  │                 │
│  │  │ N N  │ │ N N  │  │    │  │ N N  │ │ N N  │  │                 │
│  │  │ N N  │ │ N N  │  │    │  │ N N  │ │ N N  │  │                 │
│  │  └──────┘ └──────┘  │    │  └──────┘ └──────┘  │                 │
│  └─────────────────────┘    └─────────────────────┘                 │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Class Architecture

| Class           | Role                                           | Key State                                          |
|-----------------|------------------------------------------------|----------------------------------------------------|
| **Brain**       | Orchestrator — frame loop, learning, inference | frameNumber, error threshold                       |
| **Thalamus**    | Neuron registry, channel mgmt, dimension maps  | neurons Map, neuronsByValue, deathLedger, channels |
| **Memory**      | Temporal sliding window of active neurons      | activeNeurons[], inferredNeurons[], contextLength  |
| **Neuron**      | Connections, children, voting, learning, decay | connections, children, context                     |
| **Context**     | Pattern context matching & merging             | entries Map<neuron, Map<distance, strength>>       |
| **Backup**      | File persistence                               | backup/restore                                     |
| **Quantizer**   | Bucketization of continuous values to discrete | discretization                                     |
| **Diagnostics** | Debug output & accuracy tracking               | accuracy stats, mispredictions                     |

---

## Pre-Phase-3 Snapshot Test — Behavior baseline

Before any Phase 3 work, capture a behavior baseline so each implementation step can be verified against it. The goal of Phase 3 (and Phase 5 at R=1, C=1) is **bit-identical results** to today; multi-column may introduce ordering changes — see §3.15 for the discipline of resolving them by introducing stable tie-breakers rather than tolerating drift.

**Baseline = the four documented demos in [README](../README.md), run at R=1, C=1.** They cover synthetic cycles, real stock data, sequence memorization, and text learning — together they exercise pattern creation, error correction, deletion cascades, and convergence dynamics.

| # | Demo | Command | Headline numbers to verify |
|---|------|---------|----------------------------|
| 1 | [Demo 1 — Single-channel synthetic cycle](../README.md#demo-1-single-channel-synthetic-cycle) | `node apps/stocks/jobs/synthetic-extended-test.js --error-mode static --error-threshold 0.3 --merge-threshold 0.9` | Overall Optimal Rate (e.g. `233/240 = 97.1%`) |
| 2 | [Demo 3 — Stock trading, 1 episode](../README.md#demo-3-stock-trading) | `node apps/stocks/jobs/test.js` | Episode 1 net profit, total trades, base-level accuracy |
| 3 | [Demo 6 — Sequence memorization, **1 episode only**](../README.md#demo-6-stock-sequence-memorization) | `node apps/stocks/jobs/test.js --no-summary --episodes 1 --symbols KGC,GLD,SPY --context-length 3 --forget-rate 0.001 --error-mode static --error-threshold 0.3` | Episode 1 net profit, total trades, base-level accuracy |
| 4 | [Demo 7 — Text sequence learning](../README.md#demo-7-text-sequence-learning) | `node apps/text/jobs/test.js --file abramov.txt --error-mode static --error-threshold 0.3 --context-length 20 --merge-threshold 0.9 --forget-rate 0.001 --no-summary` | Per-episode accuracy across all 5 episodes |

**Verification cadence:** record the headline numbers above before starting Phase 3. Re-run after each Phase 3 implementation step (§3.1, §3.2, §3.3, §3.6, §3.7, §3.8, §3.9, §3.10, §3.11, §3.12). At R=1, C=1 every step must match the baseline exactly. Visual diff is sufficient — no programmatic snapshot file.

**Backup/restore-touching steps (§3.11) additionally verify with the [Backup → MySQL → Backup round-trip](../README.md#backup--mysql--backup-round-trip).** That test asserts a rehydrated brain reproduces the same continuation result, which is exactly what changes if save/restore drifts.

Without these baselines, drift is invisible until it's already a divergence.

---

## Phase 3 — Introduce Region and Column Classes (~1 week)

Column abstractions (Column, Region) will be introduced in JS so the data flow is fully laid out before Rust threading (Phase 5) and MPI add real parallelism. The JS preview is functionally equivalent to today's single-threaded brain — every region/column call is a synchronous local function — but the *boundaries* will become thread/process boundaries later, so they are designed to add **zero per-frame round-trips per neuron**. All cross-boundary chatter is one of the four operations enumerated in §3.5.

### Design principle: keep central state central

The minimum partitioning required for Phase 5 thread parallelism is moving the **`Neuron` instances** (compute hot path) out into Column threads. Everything else can stay where it is today.

The reasoning: in Phase 5, Rust threads will run dispatch in parallel. Workers only **read** active state during `processFrame`; the writes (`state.votes`, `state.context`, `state.threshold`, `applyContextRefUpdates`) all happen at the level barrier *after* every worker has returned. So the orchestrator (Thalamus) can serialize all writes to its own central state without locking — there is no concurrent-write contention to design around. Distributing `Memory` was a solution to a non-problem.

This makes Phase 3 substantially smaller than originally drafted. `Memory`, the death ledger, all metadata maps, the level/age indexes, and the id allocator all stay in Thalamus, exactly as today.

### Where data lives

| Data                                                                  | Owner    | Notes                                                                                          |
|-----------------------------------------------------------------------|----------|------------------------------------------------------------------------------------------------|
| `Neuron` instances                                                    | Column   | The only thing that physically moves out of Thalamus. Compute hot path.                        |
| `Memory` (`neuronStates`, `ageIndex`, `levelIndex`, `inferredNeurons`)| Thalamus | Unchanged from today. Reads packaged by Thalamus; writes serialized at the level barrier.       |
| `neuronLevels`, `neuronParents`, `baseNeurons`, `neuronsByValue`      | Thalamus | Unchanged. Thalamus uses these directly when building `levelContext` and routing cleanup.       |
| `deathLedger`, `neuronDeathFrame`                                     | Thalamus | Unchanged.                                                                                     |
| `levelCounts`                                                         | Thalamus | Unchanged.                                                                                     |
| `channelSpecs`, `dimensionSpecs`, name↔id maps                        | Thalamus | Unchanged.                                                                                     |
| `channelActions`, `actionIds`, `channelDefaultActions`                | Thalamus | Replicated read-only into each Column at init (used by `Neuron` constructors and matching).    |
| `quantizer`                                                           | Thalamus | Unchanged.                                                                                     |
| `nextNeuronId` allocator                                              | Thalamus | Unchanged.                                                                                     |
| Routing rule `(R, C)`                                                 | Thalamus | New. See §3.1.                                                                                 |

### 3.2 `Column` class

Owns `Neuron` instances and exposes a small set of batch operations on them. Becomes a Rust thread in Phase 5.

**Owned state:**
- `neurons: Map<id, Neuron>` — the Neuron objects this column owns

**Per-frame methods** (each takes a batch and returns a batch):
- `processLevel(tasks, sensoryNeurons, rewards, levelContext, newErrorPatternIds, frameNumber)` → `{results}` — see §3.6 (Op-4)
- `updateContextRefs(updates)` — see §3.10 (Op-5)
- `createNewNeurons(specs)` — see §3.9 (Op-1) and §3.8 (Op-3)
- `deleteNeurons(opBatch)` → `{outboundOps, newlyDeletableIds}` — see §3.7 (Op-2)

**Save/restore methods** (parallel across columns at init/shutdown) — see §3.11:
- `dumpAll()` → serialized neuron data for owned neurons
- `restoreNeurons(specs)` → constructs Neuron instances from serialized data

**Implementation:** create the class with the `neurons` map and method skeletons. Each method body is a thin wrapper around the equivalent code in today's Thalamus (`applyFrameResults` for the dispatch path, `cleanupPatternFromParentContext` etc. for cleanup). Sections §3.6–§3.10 refactor those code paths into clean batch operations.

### 3.3 `Region` class

Wraps `Column[C]`. In Phase 5, this is the MPI rank (the only rank, since MPI is future work). Owns no state — pure router and aggregator within its rank.

**Per-frame methods:**
- `processLevel(tasksByColumn, ...)` → fans out, collects results
- `updateContextRefs(updatesByColumn)` → fans out
- `createNewNeurons(specsByColumn)` → fans out
- `deleteNeurons(opBatchByColumn)` → fans out, collects outbound op batches and newly-deletable ids

**Implementation:** plain wrapper class. Forwards calls one-to-one to columns and concatenates results. With C=1 trivial; the structure is what matters.

### 3.4 Init-time data distribution

Immutable read-only state used inside `Neuron` constructors and matching is copied into each Column at init so Column threads have everything they need without reaching back to Thalamus:

```
Brain → Thalamus.init(R, C):
        1. Thalamus registers all channel specs (existing flow).
        2. Thalamus constructs Region[R], passing each region a copy of:
             - channelActions, actionIds, channelDefaultActions
        3. Each Region constructs Column[C], passing each column the same copy.
        4. Each Column passes the action sets into every Neuron it constructs.
```

Columns receive batches already grouped for them and emit batched results. We will thread the copies through the constructor chain in Phase 5.

### 3.5 Per-frame operations overview

There are **five** cross-region operations per frame.

| #  | Operation                  | Frequency      | Round-trips | Section |
|----|----------------------------|----------------|-------------|---------|
| 1  | Create sensory neurons     | once per frame, only when new sensory points appear | 1 (down only — fan-out) | §3.9 |
| 2  | Delete neurons             | once per frame | **variable (loop)** — bounded by cascade chain depth | §3.7 |
| 3  | Create error patterns      | once per level | 1 (down only — fan-out) | §3.8 |
| 4  | Process frame              | once per level | 1 (down + up)           | §3.6 |
| 5  | Update context references  | once per level | 1 (down only — fan-out) | §3.10 |

Per-frame call order at the brain:

```
brain.processFrame(inputs, rewards):
  buildFrame(inputs)                          // central — collects new sensory specs

  ╔═ Op-1: Create sensory neurons ══════════╗  // 1 fan-out (only if new sensory
  ╚═════════════════════════════════════════╝  //   points appeared this frame)

  ╔═ Op-2: Delete neurons (cascade pulses) ═╗
  ╚═════════════════════════════════════════╝

  ageContext(rewards)                          // Thalamus mutates Memory directly
  activateNewNeurons()                            // Thalamus mutates Memory directly

  for level = 0 .. maxActiveLevel:
    // Thalamus reads its own Memory, walks owned active neurons at level,
    // builds mergedLevelContext + correction requests + per-(neuron, age) tasks.
    // All local — no cross-region traffic.

    ╔═ Op-3: Create error patterns ════════╗  // 1 fan-out (no return)
    ╚══════════════════════════════════════╝
    ╔═ Op-4: Process frame (dispatch) ═════╗  // 1 round-trip
    ╚══════════════════════════════════════╝
    // Thalamus serially applies result writes to its own Memory:
    //   - state.votes / state.context / state.threshold per (neuron, age)
    //   - registerDeath for matched + correction patterns
    //   - activations → activatePattern (mutates Memory directly)
    //   - pre-aggregate votes for consensus
    ╔═ Op-5: Update context references ════╗  // 1 fan-out (no return)
    ╚══════════════════════════════════════╝

  determineConsensus                           // central
  saveInferredNeurons                          // central
```

> ⚠️ **Significant: level barrier.** Every Column must finish level N's dispatch before any starts N+1 (activations at N feed N+1). The level loop lives at the Brain layer; the dispatch call at each level is the synchronization point.

### 3.6 Op-4: Process frame

```
Down (thalamus → region → column):
  Per-column packages from Thalamus:
    tasks = [{neuronId, ageStates, corrections, errorFeedback}, ...]
            (Thalamus pulls ageStates from its own Memory and groups by
             owning column via the routing rule)
  Plus same-payload broadcasts:
    mergedLevelContext, newErrorPatternIds,
    sensoryNeurons (per-age, decorated with channelId/type),
    rewards, frameNumber

Compute (per column, parallel):
  For each task:
    result = neuron.processFrame(ageStates, levelContext, newErrorPatternIds,
                                 sensoryNeurons, rewards, frameNumber, ...)
  Pre-aggregate votes locally per (neuronId) and per (dimId).

Up (column → region → thalamus):
  Per-column results:
    - matches + correctionActivations (for activatePattern + registerDeath)
    - pre-aggregated votes
    - contextRefUpdates batched by target column

Thalamus, serially (no contention — workers are quiesced at the barrier):
  - writes state.votes / state.context / state.threshold to its own Memory
  - calls registerDeath for matched + correction patterns
  - calls activatePattern for each activation (mutates Memory directly)
  - merges per-region pre-aggregated votes for consensus
```

The ageStates payload on the down-trip is the only per-neuron data shipped each frame. In Phase 5 (shared-memory threads), this is essentially free (passing references). In MPI, this is the obvious target for region-side caching — see [future-work.md](future-work.md).

> Op-4 payloads are self-contained — anything `neuron.processFrame` needs is included (sensory ids decorated with `channelId`/`type`, the rewards Map, `levelContext`, etc.). No callbacks back to Thalamus during compute. Already true today; preserve in Phase 5/MPI.

**Implementation:** refactor `Thalamus.processLevel` to package per-column tasks and call `region.processLevel(...)` instead of looping `this.neurons.get(...).processFrame(...)` directly. Keep the result-write code (`applyFrameResults`) in Thalamus. Votes flow back as raw arrays (one entry per cast vote) — same shape as today, just routed through the column boundary. Pre-aggregation is a separate optional optimization in §3.12.

### 3.7 Op-2: Delete neurons (cascade in pulses)

> 🔶 **Significant: this is the only frame operation with a variable number of round-trips.**

**Why it's a loop, not a fixed count:** when pattern P dies, removing P from another pattern Q's context entries may cause Q's `canDeleteChild` to flip true, queueing Q for deletion. Q's deletion may then orphan R, etc. Cascade depth is bounded by chain length (typically 1–2 pulses, occasionally more, capped by max-level depth). Collapsing this into a fixed count would require either synchronous reads back from sender (per-neuron round-trips) or deferring cleanup to the next frame (eventual consistency — risky because dangling refs would feed into Op-4's matching).

**`deleteNeurons` op types** (the batch passed to `column.deleteNeurons`):
- `DeleteSelf(neuronId)` — this neuron is dying; walk its `routingTable` / `contextRefs` / `contextIndex` and emit outbound ops; remove the Neuron from the local map
- `RemoveFromParentRouting(parentId, childId)` — remove an entry from a live Neuron's routing table
- `RemoveContextEntry(parentId, childId, ctxNeuronId, distance)` — remove a context entry; may flip `canDeleteChild` and add to `newlyDeletableIds`
- `RemoveContextRef(ctxNeuronId, parentId, distance)` — drop a contextRef on a live Neuron
- `RemoveChildFromContexts(referencingParentId, childPatternId, distances)` — remove a referenced pattern from a Neuron's context; may flip `canDeleteChild`
- `DeleteOrphan(childId)` — child whose parent just died; treated as `DeleteSelf` on the receiver

**Pulse architecture:**

```
Setup (Thalamus, local):
  reaped = death ledger entries for this frame
  initialBatch[col] = [DeleteSelf(id) for each id owned by col]

repeat (each iteration is one round-trip):
  Per column, parallel:
    column.deleteNeurons(inbound[col]):
      For each op in inbound[col]:
        - DeleteSelf(id):
            walk Neuron's state (routingTable, contextRefs, contextIndex);
            emit outbound ops to target columns;
            remove Neuron from local map
        - other op types: apply local mutation;
            may flip canDeleteChild → add to newlyDeletableIds
      Returns {outboundOps, newlyDeletableIds}

  Thalamus collects:
    - removes deleted ids from Memory + metadata maps + death ledger
    - routes outboundOps by target column → next inbound
    - wraps newlyDeletableIds as new DeleteSelf ops → also next inbound

terminate when: every column returns empty outboundOps AND empty newlyDeletableIds
```

Each iteration = one round-trip. Every op is pure data describing a local mutation on the receiver. Thalamus removes dead ids from Memory and the metadata maps as each iteration returns.

**Implementation:** refactor `Thalamus.deletePatterns` into the pulse-loop driver. Move the per-Neuron mutations (today in `cleanupPatternFromParentContext`, `cleanupPatternFromChildContexts`, `cleanupOrphanedChildren`) into `Column.deleteNeurons` keyed by op type. Drop the dead `neuronLevels.get` argument at the `canDeleteChild` call site — the parameter isn't received by the method, and removing it eliminates the only would-be cross-column metadata lookup in the cascade.

### 3.8 Op-3: Create error patterns

> 🔶 **Significant: centralized chokepoint for id allocation.** Thalamus is the single id allocator — the natural serialization point for "what new patterns came into existence this frame".

Thalamus already has all the inputs locally:

```
Pre-step (Thalamus, local):
  Walk active neurons at this level (from Memory).
  Build mergedLevelContext directly (used later as a parameter to Op-4).
  For each (neuron, age > 0) where evaluateVoteError fires:
    correctionRequest = {parentId, age, contextEntries, connectionsSpec}
    connectionsSpec resolved using local neuronChannelId / reward lookups

Op-3 fan-out:
  Allocate a contiguous id block:
    newIds = [nextNeuronId .. nextNeuronId + count - 1]
    nextNeuronId += count
  Split specs by routing rule (per-column).

  ── Down (thalamus → region → column) ──
    For each target column, send its own specs batch (per-column payload).
    Each target column constructs its assigned new Neurons locally with the
    connection spec already in hand.
    Thalamus simultaneously updates its own metadata maps (neuronLevels,
    neuronParents, levelCounts) for the new patterns. The new patterns are
    NOT activated here — activation happens during Op-4 result writes when
    correctionActivations come back.
```

Because Thalamus has Memory and all metadata, no fan-in from columns is needed. Thalamus also assembles `newErrorPatternIds` locally — it goes down to columns as a parameter to Op-4 (along with `mergedLevelContext`), not as part of Op-3.

**Implementation:** refactor `Thalamus.createPatternNeuron` so the central call only allocates the id and resolves the connection spec. The actual `new Neuron(...)` move into `Column.createNewNeurons`. `Thalamus.processLevel`'s per-level orchestration calls `region.createNewNeurons(specsByColumn)` after building the corrections list.

### 3.9 Op-1: Create sensory neurons

When `buildFrame` quantizes a frame point into a `(dimId, bucketId)` coordinate that doesn't yet have a sensory neuron, today's code creates the Neuron inline via `addSensoryNeuron`. In the new design, Thalamus collects all new sensory specs encountered during `buildFrame`, allocates ids centrally, then fans out one `createNewNeurons` batch per owning column at the end of `buildFrame` — before the level loop starts.

Most frames create zero new sensory neurons (after warmup, the coordinate space is mostly populated). When the per-frame count is zero, the fan-out is skipped entirely.

**Implementation:** modify `buildFrame` to collect new sensory specs instead of creating inline. After the frame is built, call `region.createNewNeurons(specsByColumn)` once. Thalamus updates its own metadata maps (`neuronLevels`, `baseNeurons`, `neuronsByValue`, `levelCounts`) inline.

### 3.10 Op-5: Update context references

```
Down (thalamus → region → column):
  Thalamus routes the contextRefUpdates batch produced by Op-4 to target columns.
  Each Column applies its inbound batch to its owned Neurons.
  No return — fire-and-forget within the level barrier.
```

Batching by target column at emission time keeps this to one fan-out per level regardless of how many updates fly.

**Implementation:** refactor the `applyContextRefUpdates` loop in `applyFrameResults` to emit per-column batches and call `region.updateContextRefs(updatesByColumn)`.

### 3.11 Snapshot save and restore (parallel across columns)

Save and restore are the only operations in the system that bulk-touch every Neuron in the brain at once. Both are fully parallel across columns — one fan-in for save, one fan-out for restore. Outside of these two operations, save/load adds no cross-region traffic.

**Save** (called once at episode end / shutdown):

```
Phase 1 — Materialize lazy decay (Thalamus, local):
  thalamus.materializeAndResetNeurons(currentFrame)
  // every owned Neuron's child-strength values are walked into their
  // post-decay current values with lastActivationFrame = 0.
  // (This same call already exists today; in the new design it fans out
  // to columns since the Neuron instances live there.)

Phase 2 — Column fan-in (parallel, all columns at once):
  column.dumpAll() → [{id, neuronData}, ...] for every owned neuron
    neuronData = persistent Neuron state:
      connections, routingTable, contextRefs, contextIndex,
      patternForgetRate, ...

Phase 3 — Assemble snapshot (Thalamus, local):
  For each id collected, bundle:
    {
      id,
      neuronData,    // from column
      level,         // from thalamus.neuronLevels
      parentId,      // from thalamus.neuronParents (pattern only)
      baseNeuron,    // from thalamus.baseNeurons (sensory only)
      deathFrame     // from thalamus.neuronDeathFrame (pattern only)
    }
  Plus channelNameToId, dimensionNameToId.

Phase 4 — Write to disk (Thalamus, local):
  Existing Backup.save flow.
```

> **Note on `deathFrame`:** today's snapshot does not persist the death ledger — restore *recomputes* each pattern's death frame by reading the parent's routingTable strength. In the new design, parent and child can live on different columns, which would turn that recomputation into a per-pattern cross-column join. Persisting `deathFrame` in the snapshot avoids that join entirely. Schema bump on the backup format; small one-time migration.

**Load** (called once at startup):

```
Phase 1 — Read from disk (Thalamus, local):
  Existing Backup.loadLatest flow.

Phase 2 — Restore central state (Thalamus, local):
  - reset all Thalamus maps + Memory
  - advance id allocators past persisted max
  - restore channelNameToId, dimensionNameToId
  - for each snapshot entry, populate the central maps directly:
      neuronLevels, neuronParents, baseNeurons, neuronsByValue,
      levelCounts, deathLedger, neuronDeathFrame
  // No column traffic in this phase — Thalamus has everything it needs locally.

Phase 3 — Bucket by current routing rule (Thalamus, local):
  for each snapshot entry:
    {regionIdx, columnIdx} = thalamus.routeNeuron(entry.id)
    bucket[regionIdx][columnIdx].push(entry)
  // The CURRENT (R, C) is applied — not whatever was at save time.
  // This is how a 10-column save loads cleanly into a 20-column run.

Phase 4 — Column fan-out (parallel, all columns at once):
  column.restoreNeurons(bucket[col])
    → column constructs and stores its assigned Neuron instances locally
       from neuronData, no further coordination needed.
```

Per-column work in Phase 2/save and Phase 4/load is independent — no barriers within a phase, no cross-column messaging. Total time is `max(per-column work)` rather than sum; in MPI later, the network shape is one bulk push from each rank for save, one bulk push to each rank for load.

**Implementation:** add `dumpAll()` and `restoreNeurons(specs)` to the Column API. Bump the backup schema to persist `deathFrame` per pattern. Brain-layer save/load entry points are unchanged.

### 3.12 Vote pre-aggregation (optional optimization)

After Phase 3's correctness-preserving implementation lands and is verified bit-identical against the snapshot test (§Pre-Phase-3 Snapshot Test), pre-aggregating votes per column before they cross the region boundary is a straightforward perf win — shrinks the Op-4 up-trip payload from O(votes) to O(unique-voters + unique-dims).

**The change:** instead of returning raw vote arrays from `Column.processLevel`, each column locally folds:
- `candidates: Map<neuronId, {strength, weightedTotal}>` — sums `strength` and `strength * reward` per voted-for neuron
- `dimTotalStrength: Map<dimId, number>` — accumulates dimension-wide strength totals (events only)

These maps are merged at region, then at Thalamus, then handed to `Brain.determineConsensus` directly.

**Behavior risk:** floating-point summation isn't associative — `(a+b)+c ≠ a+(b+c)` in low-order bits. Pre-aggregating in a different order than today's flat `aggregateVotes` loop will produce slightly different vote totals. In practice the differences are tiny (rounding-noise scale), but they can flip tie-breakers in `isBetterCandidate` where two candidates' scores are very close. The snapshot test will detect any drift.

**Implementation:** modify `Column.processLevel` to fold votes locally before returning. Adapt `Region.processLevel` to merge column maps. Adapt `Thalamus`'s consensus path to skip its own re-aggregation. Then re-run the snapshot test — expect small diffs in accuracy stats; verify they're rounding-noise-scale and not material.

### 3.13 `Thalamus` changes (summary)

What moves out of Thalamus:
- `neurons: Map<id, Neuron>` — the actual Neuron instances move to Column. Thalamus keeps the metadata maps but no longer holds the Neuron objects themselves.

What stays in Thalamus, exactly as today:
- `Memory` (`neuronStates`, `ageIndex`, `levelIndex`, `inferredNeurons`)
- `deathLedger`, `neuronDeathFrame`
- `levelCounts`
- `neuronLevels`, `neuronParents`, `baseNeurons`, `neuronsByValue`
- `channelSpecs`, `dimensionSpecs`, name↔id maps
- `quantizer`, `channelActions`, `actionIds`, `channelDefaultActions`
- `nextNeuronId` allocator

What changes shape:
- `processLevel(...)` — orchestrates per-level via `region.processLevel(...)`. See §3.6.
- `deletePatterns(...)` — orchestrates the cascade pulse loop. See §3.7.
- `createPatternNeuron(...)` — id + spec resolution only; Neuron construction in Column. See §3.8.
- `addSensoryNeuron(...)` — id + spec resolution only; batched at end of `buildFrame`. See §3.9.
- `getSnapshot()` / `restoreSnapshot()` — fan-in/fan-out across columns. See §3.11.

### 3.14 Brain changes (summary)

Almost none. `Brain.processFrame` is structurally unchanged. The only differences:
- `cleanupDeadPatterns` → calls Thalamus's new pulse-loop cleanup (§3.7)
- `processLevels` → unchanged orchestration; `thalamus.processLevel` uses column-batched dispatch internally (§3.6)
- `determineConsensus` may later consume pre-aggregated `(candidates, dimTotalStrength)` maps if §3.12 is applied (perf, not correctness)

### 3.15 Behavior preservation discipline

Phase 3 must preserve today's behavior exactly — at R=1, C=1 *and* at R>1 / C>1. The plan is **not** to tolerate drift: when an implementation step changes an iteration order or aggregation order, the fix is to introduce a stable tie-breaker / sort key that reproduces today's order on the new code path. Each such tie-breaker is recorded here as it is discovered, so the set of ordering rules the system relies on is explicit.

**Working loop.** After each implementation step, re-run the four baselines (§Pre-Phase-3 Snapshot Test). If a number changes:
1. Find the iteration / aggregation site whose order changed.
2. Identify the order today's code happened to produce.
3. Add an explicit sort or stable key on the new code path that reproduces it.
4. Document the tie-breaker in the table below.
5. Re-run the baselines — must match exactly before moving on.

**Known tie-breakers (today's behavior already depends on these; preserve them):**
- `isBetterCandidate` breaks ties by `neuronId < best.neuronId` — order-independent, no action needed.

**Tie-breakers introduced during Phase 3:** *(none yet — populated as drift surfaces)*

| # | Site | Order today | Stable key applied |
|---|------|-------------|--------------------|
| — | — | — | — |

**Phase 5 / MPI extension of the same discipline.** When collecting per-thread results in the Rust core, concatenate by column index (not arrival order) so results are deterministic regardless of thread scheduling. Same discipline applies in MPI later.

---

## Phase 4 — Single-Threaded Rust Core + Node.js Bindings (~2 weeks)

Rewrite the core brain computation in Rust as a single-threaded library. Publish as an npm package with N-API bindings so the existing Node.js app can call into it. Extra week budgeted for Rust learning curve (ownership/borrow checker will fight neuron graph patterns).

### 4.1 Set up Rust project with N-API bindings
- Cargo workspace with `brain-core` library crate
- N-API bindings via `neon` or `napi-rs` for Node.js interop
- Replicate Neuron, Context, Connection data structures in Rust

### 4.2 Implement core brain logic in Rust (single-threaded)
- Neuron struct, Context struct, connection maps — all ID-based
- `process_frame()` per neuron: recognition, connections, pattern learning, voting
- `process_levels()` loop: iterates levels, calls `process_frame()` on each active neuron
- Serialize/deserialize for dump backup/restore

### 4.3 Rust unit tests
- Mirror JS test scenarios as native Rust tests (faster to debug than FFI round-trip)
- Frame-level verification: given same inputs, Rust produces same votes/patterns/connections as JS
- Serialize/deserialize round-trip tests

### 4.4 Dump cross-compatibility testing
- Rust must load JS-produced dumps and produce identical results
- JS must load Rust-produced dumps and produce identical results
- Watch for: floating-point rounding differences, map iteration order, ID assignment sequences

### 4.5 Wire Rust core into JS Brain via N-API
- Brain calls into Rust for compute-heavy frame processing
- JS retains: channel I/O, database analysis, diagnostics
- Rust returns: aggregated votes, new patterns, connection updates

### 4.6 Publish npm package
- `brain-core` available as native addon via npm
- Prebuilt binaries for common platforms (Windows, Linux, macOS)
- JS wrapper remains the user-facing API

---

## Phase 5 — Multi-Threaded Rust Core + Column Classes (~1 week)

Add threading within the Rust core. Introduce the region/column abstractions directly in Rust (design from Phase 2, implemented here).

### 5.1 Implement `Column` in Rust
- Owns a partition of neurons (arena-allocated)
- Calls `neuron.process_frame()` on each owned neuron
- No locks needed — single-owner per thread

### 5.2 Implement `Region` with thread pool
- Spawns N worker threads, each running a `Column`
- Uses channels (crossbeam) or shared memory for intra-column communication
- Barrier synchronization at frame boundaries
- Vote aggregation across columns

### 5.3 Neuron metadata storage for multi-threaded
- Each Region holds the metadata lookup tables (channel, type, dimension, value, level, parentId) for its owned neurons (dimension/value are level-0 only)
- Columns within the same Region read from the parent region's tables via shared memory — no copies needed
- All neuron metadata is immutable after creation, so no synchronization is required for reads
- Lookup interface remains the same as single-threaded (`get_channel(neuron_id)`, `get_type(neuron_id)`, etc.)

### 5.4 Neuron-to-thread ownership map
- Maintain a 2-way map: `neuronId ↔ threadId/columnId`
- When a neuron is created, Brain decides which thread/column owns it
- This map is used for routing: when a neuron references a foreign ID, the system knows which thread to query

### 5.5 Neuron distribution on load, collection on save
- When loading from a dump, neurons are distributed to the thread/column pool based on partitioning strategy
- When saving, Brain collects neuron state from each thread/column and serializes centrally

### 5.6 Neuron partitioning strategy
- All neurons (sensory and pattern) partitioned by mod over `neuronId` per the routing rule from Phase 3 §3.1: `regionIdx = neuronId % R`, `columnIdx = floor(neuronId / R) % C`
- No special-casing for sensory vs pattern, and no parent-co-location — every neuron's owner is purely a function of its id
- No dynamic rebalancing planned; revisit only if profiling surfaces a hot path

---

## Phase 6 — Scale Stock Processing (~1 week)

With multi-threaded Rust core in place, focus on scaling the stock trading workload as the primary benchmark.

### 6.1 Multi-stock parallel processing
- Multiple stock channels processed in parallel across columns
- Benchmark: throughput vs single-threaded JS baseline

### 6.2 Performance tuning
- Inverted index for pattern recognition already implemented in JS (Step 1.1b) and ported to Rust (Phase 4) — validate selectivity metrics at scale
- Profile hot paths (vote aggregation, connection updates, index maintenance)
- Optimize memory layout for cache locality (arena allocation, struct-of-arrays where beneficial)
- Tune thread count and neuron partitioning for stock workloads

### 6.3 Add pre-training demo
- Change offset rows to work from the end and add demo for training first and then testing accuracy with hold out data

---

## Key Design Decisions

Resolved by the Phase 3 design above:
- **Neuron ownership granularity**: fixed partition by `neuronId` mod `(R, C)`; no dynamic rebalancing (§3.1, §5.6).
- **Cross-column connections**: connections are id-only; metadata is decorated onto the Op-4 payloads (§3.6); no proxy neurons.
- **Synchronization model**: lock-step per level. Workers only read during dispatch; Thalamus serializes all state writes at the level barrier — no locks needed.
- **Memory/context scope**: `Memory` stays central in Thalamus. `Neuron` instances move out to Columns. Thalamus reads Memory locally to build `levelContext` and packages per-column ageStates for Op-4.
- **Pattern creation across boundaries**: Thalamus allocates the id and resolves the connection spec from its own metadata; Op-3 broadcasts the install batch down for the owning Column to construct (§3.5 Op-3).
- **Channel assignment**: all channels visible to all columns via init-time replication of action sets (§3.2).

Still open:
- **Serialization format** — what binary format for dumps? msgpack, protobuf, custom?

---

## Mapping: Current JS → Target Architecture

```
Current JS                          Target Rust + MPI
─────────────────────────────────   ─────────────────────────────────
Brain                         →     Brain (thin JS coordinator)
                                      ├── Region (MPI rank 0)
                                      │     ├── Column thread 0
                                      │     │     ├── Neuron pool
                                      │     │     ├── Local memory
                                      │     │     └── Local pattern recognition
                                      │     ├── Column thread 1
                                      │     └── Vote aggregator
                                      ├── Region (MPI rank 1)
                                      └── Global consensus (MPI)

Thalamus                      →     Split: per-column registry + global lookup
Memory                        →     Split: per-column window + region aggregator
Neuron                        →     Rust struct (arena-allocated, column-owned)
Context                       →     Rust struct (inline with neuron)
Channel                       →     Stays in JS (I/O boundary)
Backup (CSV files)            →     Owned by Rust core (mirrors DB schema)
apps/db (MySQL utilities)     →     Stays in JS, outside brain core (analysis only)
```

---

## Timeline Summary

| Phase | Scope | Estimate |
|-------|-------|----------|
| 3 | Column class design | ~1 week  |
| 4 | Single-threaded Rust core + Node.js/npm | ~2 weeks |
| 5 | Multi-threaded Rust core + column classes | ~1 week  |
| 6 | Scale stock processing | ~1 week  |

**Total: ~5 weeks**

See [future-work.md](future-work.md) for Python bindings, MPI distribution, text/vision/audio channels, robotics, and other longer-term plans.