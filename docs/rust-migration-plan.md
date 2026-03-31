# Rust Core Migration Plan

## Goal

Migrate the brain's core computation from single-threaded JavaScript to a Rust core with:
- **MPI processes** → cortical columns (inter-process communication)
- **Threads per MPI process** → mini cortical columns (shared-memory parallelism)
- **Neurons** → owned by mini cortical columns at various hierarchy levels

```
┌─────────────────────────────────────────────────────────────────────┐
│                        MPI Cluster                                  │
│                                                                     │
│  ┌─────────────────────┐    ┌─────────────────────┐                │
│  │ Cortical Column 0   │    │ Cortical Column 1   │   ...          │
│  │ (MPI Rank 0)        │    │ (MPI Rank 1)        │                │
│  │                     │    │                     │                │
│  │  ┌──────┐ ┌──────┐  │    │  ┌──────┐ ┌──────┐  │                │
│  │  │Mini  │ │Mini  │  │    │  │Mini  │ │Mini  │  │                │
│  │  │Col 0 │ │Col 1 │  │    │  │Col 0 │ │Col 1 │  │                │
│  │  │      │ │      │  │    │  │      │ │      │  │                │
│  │  │ N N  │ │ N N  │  │    │  │ N N  │ │ N N  │  │                │
│  │  │ N N  │ │ N N  │  │    │  │ N N  │ │ N N  │  │                │
│  │  └──────┘ └──────┘  │    │  └──────┘ └──────┘  │                │
│  │  ┌──────┐ ┌──────┐  │    │  ┌──────┐ ┌──────┐  │                │
│  │  │Mini  │ │Mini  │  │    │  │Mini  │ │Mini  │  │                │
│  │  │Col 2 │ │Col 3 │  │    │  │Col 2 │ │Col 3 │  │                │
│  │  │      │ │      │  │    │  │      │ │      │  │                │
│  │  │ N N  │ │ N N  │  │    │  │ N N  │ │ N N  │  │                │
│  │  │ N N  │ │ N N  │  │    │  │ N N  │ │ N N  │  │                │
│  │  └──────┘ └──────┘  │    │  └──────┘ └──────┘  │                │
│  └─────────────────────┘    └─────────────────────┘                │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Current Architecture (JS)

| Class | Role | Key State |
|-------|------|-----------|
| **Brain** | Orchestrator — frame loop, learning, inference | frameNumber, maxLevels, error threshold |
| **Thalamus** | Neuron registry, channel mgmt, dimension maps | neurons Map, neuronsByValue, deathLedger, channels |
| **Memory** | Temporal sliding window of active neurons | activeNeurons[], inferredNeurons[], contextLength |
| **Neuron** | Connections, children, voting, learning, decay | connections, children, context, coordinates |
| **Context** | Pattern context matching & merging | entries Map<neuron, Map<distance, strength>> |
| **Channel** | I/O interface (stock, text, vision, etc.) | dimensions, actions, rewards |
| **Database** | MySQL persistence | connection, backup/restore |
| **Diagnostics** | Debug output & accuracy tracking | accuracy stats, mispredictions |

---

## Phase 0 — Centralize Hyperparameters in Brain Constructor (~2 days)

All hyperparameters are currently scattered as static class fields and constructor hardcodes across Neuron, Context, Memory, and Brain. Move them all into the Brain constructor options so they can be set from the command line. This clarifies the Brain interface — which is what Rust will eventually mirror as its public API.

### Current hyperparameter locations

| Class | Parameter | Default | Purpose |
|---|---|---|---|
| Brain | `maxLevels` | 150 | Recursion limit for pattern hierarchy |
| Brain | `errorCorrectionThreshold` | 0.55 | Prediction error threshold for creating patterns |
| Neuron | `maxStrength` | 100 | Strength cap for connections/activation |
| Neuron | `minStrength` | 0 | Strength floor |
| Neuron | `rewardSmoothing` | 0.1 | Reward exponential moving average factor |
| Neuron | `connectionForgetRate` | 0.01 | Connection lazy decay rate per frame |
| Neuron | `patternForgetRate` | 0.01 | Pattern activation decay rate per frame |
| Context | `maxStrength` | 100 | Context entry strength cap |
| Context | `minStrength` | 0 | Context entry strength floor |
| Context | `mergeThreshold` | 0.5 | Threshold for pattern context matching |
| Context | `negativeReinforcement` | 0.1 | Weakening rate for missing context entries |
| Memory | `contextLength` | 10 | Sliding window size (frames) |

### Implementation

#### Step 0.1 — Test removing min/max strength caps
- Run 10 batches of 10-stock jobs with current min/max strength settings vs uncapped
- Compare accuracy, neuron counts, connection counts, pattern counts
- If no meaningful impact, remove `maxStrength` and `minStrength` from both Neuron and Context — fewer hyperparameters to carry forward into Rust
- If there is impact, keep them but document why they matter

#### Step 0.2 — Accept hyperparameters in Brain constructor options
- Brain constructor takes an optional `hyperparameters` object (or flat keys) in `options`
- Defaults match current hardcoded values — zero behavior change
- Brain distributes values to Neuron, Context, Memory on construction

#### Step 0.3 — Remove static fields, pass through constructor chain
- Neuron, Context, Memory receive their hyperparameters via constructor or init method
- Remove static class fields — they become instance-scoped (or module-scoped set once by Brain)
- This eliminates hidden global state and makes the dependency explicit

#### Step 0.4 — Wire command line options and update README
- Add `--max-levels`, `--context-length`, `--forget-rate`, etc. to the job runner CLI
- Job passes them through to Brain constructor options
- Update README demo sections to show how to run with custom hyperparameters
- **Verify**: default values produce identical results, all tests pass

---

## Phase 1 — Unify Per-Neuron Processing (~1.5 weeks)

Collapse the 4 separate passes over neurons (recognize, learn connections, learn patterns, infer) into a single per-neuron `processFrame()` call — the prerequisite for parallelization. This must be done incrementally, one operation at a time, verifying results after each step.

### Current Iteration Patterns (What We're Merging)

The 4 operations currently iterate active neurons differently:

| Operation | Iterates Over | Grouped By | Key Inputs |
|-----------|--------------|------------|------------|
| **recognizePatterns** | neurons at each level, all ages | level (ascending) | same-level context at older ages |
| **updateConnections** | all neurons at age > 0 | flat (all levels) | newly active sensory neurons (age=0) |
| **learnNewPatterns** | neurons that voted (age > 0) | flat (all levels) | previous frame votes, actual events |
| **collectVotes** | all neurons age 0..contextLength-2 | flat (all levels) | per-age/level context |

**Target**: all 4 operations merge into the level-by-level loop that recognition already uses. At each level, for each neuron, do: recognize → learn connections → learn patterns → cast votes.

**What stays global (never per-neuron)**: consensus determination, action execution, ensuring channel actions, sensor activation, cleanup.

### Dependencies Between Operations (Within a Single Frame)

```
recognizePatterns ──► sets activatedPattern flag on parent neurons
                 ──► activates pattern neurons in memory (become available at same age)
                      │
updateConnections     │ (independent — uses age=0 sensory neurons, available before loop)
                      │
learnNewPatterns      │ (independent — uses PREVIOUS frame's votes, already stored in memory)
                      │
collectVotes ◄────────┘ (depends on activatedPattern flag — suppresses parent if pattern matched)
```

Key insight: within a single neuron, the ordering is natural:
1. Try to recognize a pattern (sets activatedPattern on self)
2. Learn connections (independent)
3. Learn from errors (uses previous frame's votes on self)
4. Cast votes (skipped if activatedPattern was set in step 1)

### Incremental Steps

Each step is a self-contained refactor. Run ALL tests after each step. Results must be identical.

#### Step 1.0 — Convert all neuron references from pointers to IDs

**Why first**: Every data structure in the system uses object pointers (`Map<Neuron, ...>`, `Set<Neuron>`, direct neuron references). In a distributed system, neurons on different threads/machines can only reference each other by numeric ID. This is the most fundamental change and everything else depends on it.

**Pointer-based fields to convert**:

| Location | Field | Current | Target |
|---|---|---|---|
| Neuron | `connections` | `Map<dist, Map<Neuron, ...>>` | `Map<dist, Map<neuronId, ...>>` |
| Neuron | `children` | `Set<Neuron>` | `Set<neuronId>` |
| Neuron | `parent` | `Neuron` | `neuronId` |
| Neuron | `contextRefs` | `Map<Neuron, Set<dist>>` | `Map<neuronId, Set<dist>>` |
| Context | `entries` | `Map<Neuron, Map<dist, str>>` | `Map<neuronId, Map<dist, str>>` |
| Memory | `activeNeurons` | `Array<Map<Neuron, state>>` | `Array<Map<neuronId, state>>` |

**All callers that dereference neurons from these maps must go through Thalamus (or a passed-in lookup) to resolve ID → Neuron when they need the actual object** (e.g., to read `neuron.coordinates`, `neuron.type`, `neuron.channel`). Most internal processing (connections, context matching, voting) only needs the ID and stored numeric data — no object dereference required.

**Implementation**: convert one field at a time, fix all callers, verify tests after each sub-step:
1. `connections` — largest surface area, most callers
2. `children` and `parent`
3. `contextRefs`
4. `Context.entries`
5. `Memory.activeNeurons` — ~15 methods read this assuming Neuron objects, all need updating

**This step alone is ~3–4 days** given the surface area.

**Verify**: all tests pass after each sub-step, results identical

#### Step 1.0b — Move neuron metadata to Thalamus

Neurons currently store metadata they never use internally (`channel`, `type`, `coordinates`, `parent`). This metadata is only read by Brain, Memory, and Thalamus for routing/filtering/consensus decisions. Moving it to Thalamus makes neurons pure data processors — they only store learned associations and do pattern matching/voting based on numeric IDs and strengths.

**Fields to move off Neuron → Thalamus lookup tables**:

| Field | Neuron uses internally? | Who reads it? | Thalamus storage |
|---|---|---|---|
| `channel` | never | Brain (consensus, rewards), Memory (action grouping) | `neuronId → channel` |
| `type` | never | Brain (event vs action filtering), Memory (same) | `neuronId → type` |
| `coordinates` | only for `valueKey` (already a Thalamus lookup) | Brain (consensus dimensions, winner building) | `neuronId → coordinates` (already in `neuronsByValue`) |
| `parent` | never | Brain (error correction only) | `childId → parentId` |
| `level` | vote weighting, death frame, deletion guard | Brain/Memory (level loop, context keys, filtering) | `neuronId → level` + `level → Set<neuronId>` |

**For `level`**: the 3 internal uses (vote weighting, death frame calculation, deletion guard) all become parameters passed in by the caller. Thalamus already needs level→neuron mapping for the level loop.

**Implementation**: move one field at a time, update all callers to go through Thalamus lookups, verify after each:
1. `channel` and `type` (always accessed together — move as a pair)
2. `coordinates` (drop `valueKey` getter from Neuron)
3. `parent` (replace with Thalamus `childId → parentId` map)
4. `level` (refactor `vote()`, `strengthenActivation()`, `canBeDeleted()` to take level as parameter)

**Verify**: all tests pass after each sub-step, results identical

---

#### Step 1.1 — Move pattern contexts from children to parent routing table

**Why first**: Currently `matchPattern()` on the parent reaches into each child's `pattern.context` to match. In a distributed system, children may live on different machines. The parent must own the routing table with all context data so matching is local.

**Current flow**:
- Child pattern neuron owns `this.context` (a Context object)
- Parent's `matchPattern(observed)` iterates `this.children`, calls `pattern.context.match(observed, decay)` on each
- `refineContext()` is called on the child: `pattern.refineContext(common, novel, missing)`
- `addPatternContext()` is called on the child, which also sets `contextRefs` on the context neurons

**Target flow**:
- Parent neuron owns a routing table: `Map<patternNeuron, Context>` (or similar structure inside existing children set)
- `matchPattern()` reads from the parent's own routing table — no reaching into children
- `refineContext()` updates the parent's routing table entry for that child
- `addPatternContext()` adds entries to the parent's routing table
- Children no longer own context — they remain lightweight (just connections + children of their own)
- `contextRefs` still needed for cleanup (when a context neuron dies, find all routing tables that reference it)
- **Subtle change**: `contextRefs` currently points back to the *child pattern* that references a context neuron. When context moves to the parent's routing table, `contextRefs` must point to the *parent* instead. This changes the cleanup path in `Thalamus.deletePatterns` — must be updated carefully.

**Implementation**: small incremental sub-steps within this milestone — move context storage first, then matching, then refinement, then contextRefs cleanup path, verifying after each.

**Verify**: pattern recognition results identical, all tests pass

#### Step 1.2 — Move `updateConnections` into the level loop

- Currently `updateConnections()` iterates `memory.getContextNeurons()` (all age>0, all levels)
- Restructure: within `recognizeLevel(level)`, after pattern matching, also call `learnConnections` on neurons at this level
- The set of neurons iterated is the same — just reorganized by level
- Remove `updateConnections()` from Brain once fully merged
- **Verify**: connection learning results identical, all tests pass

#### Step 1.3 — Move `learnNewPatterns` into the level loop

- Currently `learnNewPatterns()` iterates `memory.getVotersWithContext()` (age>0, have votes from previous frame)
- Brain determines which neurons need error correction (by checking previous-frame votes against actual events)
- Brain allocates new pattern neurons via Thalamus (in bulk, per level — future: parallel allocation across machines)
- Then in the level loop, neurons that need correction are called with the new pattern ID and context
- The neuron stores the context in the parent's routing table (from Step 1.1)
- When a neuron gets a new error correction pattern activated, it should no longer vote (same as recognition suppression)
- `getActualEvents()` and `sensoryNeurons` are computed once before the loop (unchanged)
- **Verify**: pattern creation identical, all tests pass

#### Step 1.4 — Move `collectVotes` into the level loop

- Currently `collectVotes()` iterates `memory.getVotingNeurons()` (all ages 0..N-1, all levels)
- Restructure: within the level loop, after connections and pattern learning, collect votes from neurons at this level
- Votes accumulate into an array passed to `determineConsensus()` unchanged
- Suppression: neurons skip voting if they had a pattern activated — either from recognition (Step 1.2 ordering) or from error correction (Step 1.3)
- **Verify**: votes identical, inference results identical, all tests pass

#### Step 1.5 — Rename the unified loop

- `recognizeLevel()` is now doing all 4 operations — rename to `processLevel()`
- `recognizePatterns()` (the outer level loop) becomes `processLevels()`
- Brain's `processFrame()` calls `processLevels()` instead of 4 separate methods
- Remove the now-dead `updateConnections()`, `learnNewPatterns()`, `inferNeurons()`, `collectVotes()` from Brain
- Consensus determination and action execution remain in Brain, called after `processLevels()` returns votes
- **Verify**: all tests pass, behavior identical

#### Step 1.6 — Push the unified loop into Thalamus

- Brain currently owns `processLevels()` — move it to Thalamus
- Brain calls `thalamus.processFrame(...)` which returns votes + new patterns
- Brain still handles: I/O, sensor activation, consensus, action execution, cleanup
- This is the final structural separation: Thalamus owns neuron iteration, Brain owns coordination
- **Verify**: all tests pass, behavior identical

---

## Phase 2 — Introduce Column Classes (deferred to Rust — Phase 5)

Column abstractions (MiniCorticalColumn, CorticalColumn) will be introduced directly in Rust rather than building them in JS only to throw them away 2–3 weeks later. The JS refactoring in Phase 1 already validates that per-neuron processing is clean enough to parallelize. The column classes are a Rust-native concern.

The following describes the *design* for reference — implementation happens in Phase 5.

### 2.1 Introduce `MiniCorticalColumn` class
- Owns a subset of neurons
- Calls `neuron.processFrame()` on each of its neurons
- Holds per-column local memory (active neuron window for its neurons)
- Methods: `processFrame()` — iterates owned neurons, returns aggregated results

### 2.2 Introduce `CorticalColumn` class
- Wraps a set of `MiniCorticalColumn` instances
- Owns shared state: dimension maps, channel actions
- Aggregates votes across its mini columns
- Methods: `processFrame()`, `aggregateVotes()`, `distributeInputs()`

### 2.3 Refactor Brain as top-level coordinator
- Brain becomes a thin coordinator over `CorticalColumn` instances
- Brain handles: I/O (channels), global consensus across columns, action execution
- Brain no longer directly touches individual neurons

### 2.4 Refactor Thalamus for column-aware neuron ownership
- Neurons get assigned to a specific mini column (owner)
- Thalamus tracks which column/mini-column owns each neuron
- Neuron lookup still global (Thalamus), but mutations route through owner

### 2.5 Refactor Memory for per-column active state
- Each mini column has its own active neuron window
- Global Memory becomes an aggregator over per-column memories
- Inferred neurons remain global (consensus output)

---

## Phase 3 — Clean Up Persistence (~1 week)

Clarify the two distinct persistence concerns before moving to Rust, so the library boundary is clean.

**Role clarification**:
- **Dumps** = backup/restore. Serialize entire brain state to/from a portable byte format. Used for saving on exit, loading on startup, checkpointing.
- **Database** = debugging/analysis. Indexed, queryable representation of neuron relationships for "brain deep dive" tools. Not for backup/restore.

**Responsibility split**:
- **Rust core (library)**: owns `serialize() → bytes` and `deserialize(bytes) → brain state`. The library knows its internal data structures — only it can produce a correct serialization. No file I/O, no database, no external dependencies.
- **App/wrapper (JS or future bindings)**: decides *where* and *when* to persist. Calls `core.serialize()`, writes to file. Reads file, calls `core.deserialize(bytes)`. Also owns database population for analysis tools (iterates neurons via core query APIs, writes to indexed tables).

### 3.1 Convert dumps to the primary backup/restore mechanism
- Dumps become binary serialization of full brain state (neurons, connections, contexts, routing tables)
- Brain constructor options: `--dump-file <path>` to save on exit, `--load-dump <path>` to restore on startup
- Remove backup/restore from Database class — it becomes analysis-only

### 3.2 Refactor Database to analysis-only
- Database no longer handles backup/restore
- Database writes indexed neuron/connection/context data for query tools
- Optional: populate on demand (e.g., `--analyze` flag) rather than on every shutdown
- Brain constructor options: `--database` enables analysis writes, no longer implies backup

### 3.3 Prepare serialization interface for Rust migration
- The JS dump implementation becomes the reference for the Rust `serialize`/`deserialize` API
- Format should be portable (not JS-specific) — binary or msgpack, not JSON
- Version the format so Rust core can evolve internals without breaking saved dumps

---

## Phase 4 — Single-Threaded Rust Core + Node.js Bindings (~3 weeks)

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

Add threading within the Rust core. Introduce the column/mini-column abstractions directly in Rust (design from Phase 2, implemented here).

### 5.1 Implement `MiniCorticalColumn` in Rust
- Owns a partition of neurons (arena-allocated)
- Calls `neuron.process_frame()` on each owned neuron
- No locks needed — single-owner per thread

### 5.2 Implement `CorticalColumn` with thread pool
- Spawns N worker threads, each running a `MiniCorticalColumn`
- Uses channels (crossbeam) or shared memory for intra-column communication
- Barrier synchronization at frame boundaries
- Vote aggregation across mini columns

### 5.3 Neuron partitioning strategy
- Sensory neurons partitioned by channel/dimension hash
- Pattern neurons live on same mini column as parent
- Dynamic rebalancing deferred to Phase 8 (MPI)

---

## Phase 6 — Scale Stock Processing (~1 week)

With multi-threaded Rust core in place, focus on scaling the stock trading workload as the primary benchmark.

### 6.1 Multi-stock parallel processing
- Multiple stock channels processed in parallel across mini columns
- Benchmark: throughput vs single-threaded JS baseline

### 6.2 Performance tuning
- Profile hot paths (pattern matching, vote aggregation, connection updates)
- Optimize memory layout for cache locality (arena allocation, struct-of-arrays where beneficial)
- Tune thread count and neuron partitioning for stock workloads

---

## Phase 7 — Python Bindings + PyPI (optional, ~1 week)

Expose the Rust core to Python for broader adoption.

### 7.1 Python bindings via PyO3/maturin
- Wrap `brain-core` Rust library with PyO3
- Pythonic API matching the Node.js wrapper patterns
- Build and publish to PyPI via maturin

### 7.2 Python channel interface
- Python equivalent of Channel base class
- Example: stock channel in Python

---

## Phase 8 — MPI Distribution (optional, future)

Distribute across multiple machines for large-scale workloads.

### 8.1 Add MPI layer for inter-column communication
- Each MPI rank runs one `CorticalColumn`
- MPI messages: vote broadcasts, neuron migration, consensus sync
- Use `rsmpi` crate for Rust MPI bindings

### 8.2 Global consensus protocol
- Each column produces local vote aggregation
- MPI AllReduce or custom gather for global consensus
- Actions executed by rank 0 (or designated I/O rank)

### 8.3 Neuron partitioning and migration
- Sensory neurons assigned to columns by channel/dimension hash
- Pattern neurons live on same column as parent
- Migration protocol for rebalancing load across columns

---

## Key Design Decisions (To Refine)

1. **Neuron ownership granularity** — how many neurons per mini column? Fixed partition or dynamic?
2. **Cross-column connections** — neurons in different columns can have connections. How are these resolved? Proxy neurons? Message passing?
3. **Synchronization model** — lock-step frame processing across all columns, or async with eventual consistency?
4. **Memory/context scope** — is the active neuron window per-mini-column, per-column, or global?
5. **Pattern creation across boundaries** — what happens when a pattern's parent and context neurons are on different columns?
6. **Channel assignment** — one channel per column, or all channels visible to all columns?
7. **Serialization format** — what binary format for dumps? msgpack, protobuf, custom?

---

## Mapping: Current JS → Target Architecture

```
Current JS                          Target Rust + MPI
─────────────────────────────────   ─────────────────────────────────
Brain                         →     Brain (thin JS coordinator)
                                      ├── CorticalColumn (MPI rank 0)
                                      │     ├── MiniCol thread 0
                                      │     │     ├── Neuron pool
                                      │     │     ├── Local memory
                                      │     │     └── Local pattern recognition
                                      │     ├── MiniCol thread 1
                                      │     └── Vote aggregator
                                      ├── CorticalColumn (MPI rank 1)
                                      └── Global consensus (MPI)

Thalamus                      →     Split: per-column registry + global lookup
Memory                        →     Split: per-mini-col window + column aggregator
Neuron                        →     Rust struct (arena-allocated, column-owned)
Context                       →     Rust struct (inline with neuron)
Channel                       →     Stays in JS (I/O boundary)
Database                      →     Stays in JS (persistence boundary)
```

---

## Timeline Summary

| Phase | Scope | Estimate |
|-------|-------|----------|
| 0 | Centralize hyperparameters in Brain constructor | ~2 days |
| 1 | Unify per-neuron processing (JS refactor) | ~1.5 weeks |
| 2 | Column class design (deferred to Rust — Phase 5) | — |
| 3 | Clean up persistence (dumps/database) | ~1 week |
| 4 | Single-threaded Rust core + Node.js/npm | ~3 weeks |
| 5 | Multi-threaded Rust core + column classes | ~1 week |
| 6 | Scale stock processing | ~1 week |
| 7 | Python bindings + PyPI | optional |
| 8 | MPI distribution | optional |

**Total core work (Phases 0–6):** ~8 weeks
**Optional (Phases 7–8):** as needed

---

## Success Criteria

- **Phase 0**: All hyperparameters configurable via Brain constructor and CLI. Default values produce identical results. No static class fields for hyperparameters.
- **Phase 1**: All existing tests pass identically. Brain behavior unchanged. Every neuron's frame work goes through a single `processFrame()` call. Brain delegates to Thalamus, not to individual neuron loops. All neuron references are ID-based. Contexts live in parent routing tables.
- **Phase 2**: Column class design documented. Implementation deferred to Phase 5 in Rust.
- **Phase 3**: Dumps are the primary backup/restore mechanism. Database is analysis-only. Serialization format is portable and versioned, ready for Rust core to own.
- **Phase 4**: Single-threaded Rust core handles frame processing. Rust unit tests pass. Dump cross-compatibility verified (JS↔Rust). JS tests pass through N-API. Published to npm. Results identical to JS implementation.
- **Phase 5**: Multi-threaded Rust core with column classes. Measurable speedup over single-threaded. Thread count configurable. Neurons partitioned across mini columns.
- **Phase 6**: Stock processing scales with parallelism. Benchmarked against JS baseline.
- **Phase 7** (optional): Python bindings published to PyPI. Python channels work end-to-end.
- **Phase 8** (optional): Multiple MPI ranks process frames in parallel. Linear scaling for neuron-heavy workloads.

