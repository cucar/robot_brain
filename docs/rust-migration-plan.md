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

## Current Architecture (JS)

| Class | Role | Key State |
|-------|------|-----------|
| **Brain** | Orchestrator — frame loop, learning, inference | frameNumber, error threshold |
| **Thalamus** | Neuron registry, channel mgmt, dimension maps | neurons Map, neuronsByValue, deathLedger, channels |
| **Memory** | Temporal sliding window of active neurons | activeNeurons[], inferredNeurons[], contextLength |
| **Neuron** | Connections, children, voting, learning, decay | connections, children, context (coordinate held in Thalamus.baseNeurons for level-0 only) |
| **Context** | Pattern context matching & merging | entries Map<neuron, Map<distance, strength>> |
| **Channel** | I/O interface (stock, text, vision, etc.) | dimensions, actions, rewards |
| **Database** | MySQL persistence | connection, backup/restore |
| **Diagnostics** | Debug output & accuracy tracking | accuracy stats, mispredictions |

---

## Phase 2 — Introduce Column Classes (deferred to Rust — Phase 5)

Column abstractions (Column, Region) will be introduced directly in Rust rather than building them in JS only to throw them away 2–3 weeks later. The column classes are a Rust-native concern.

The following describes the *design* for reference — implementation happens in Phase 5.

### 2.1 Introduce `Column` class
- Owns a subset of neurons
- Calls `neuron.processFrame()` on each of its neurons
- Holds per-column local memory (active neuron window for its neurons)
- Methods: `processFrame()` — iterates owned neurons, returns aggregated results

### 2.2 Introduce `Region` class
- Wraps a set of `Column` instances
- Owns shared state: dimension maps, channel actions
- Aggregates votes across its columns
- Methods: `processFrame()`, `aggregateVotes()`, `distributeInputs()`

### 2.3 Refactor Brain as top-level coordinator
- Brain becomes a thin coordinator over `Region` instances
- Brain handles: I/O (channels), global consensus across columns, action execution
- Brain no longer directly touches individual neurons

### 2.4 Refactor Thalamus for column-aware neuron ownership
- Neurons get assigned to a specific column (owner)
- ~~Neuron channel, type, coordinates belong to base level only - may be ok to merge as baseNeurons?~~ **Done**: Thalamus now holds a single `baseNeurons` Map (`neuronId → {channel, type, coordinate}`) for level-0 only; interneurons have no coordinate. DB `coordinates` table dropped, columns merged into `base_neurons`.
- Thalamus tracks which region/column owns each neuron
- Neuron lookup still global (Thalamus), but mutations route through owner

### 2.5 Refactor Memory for per-column active state
- Each column has its own active neuron window
- Global Memory becomes an aggregator over per-column memories
- Inferred neurons remain global (consensus output)

### 2.6 Cleanup
- Change offset rows to work from the end and add demo for training first and then testing accuracy with hold out data

---

## Phase 3 — Clean Up Persistence (~1 week)

Clarify the two distinct persistence concerns before moving to Rust, so the library boundary is clean.

**Role clarification**:
- **Dumps** = backup/restore. Serialize entire brain state to/from a portable byte format. Used for saving on exit, loading on startup, checkpointing.
- **Database** = debugging/analysis. Indexed, queryable representation of neuron relationships for "brain deep dive" tools. Not for backup/restore.

**Responsibility split**:
- **Rust core (library)**: owns `serialize() → bytes` and `deserialize(bytes) → brain state`. The library knows its internal data structures — only it can produce a correct serialization. No file I/O, no database, no external dependencies.
- **App/wrapper (JS or future bindings)**: decides *where* and *when* to persist. Calls `core.serialize()`, writes to file. Reads file, calls `core.deserialize(bytes)`. Also owns database population for analysis tools (iterates neurons via core query APIs, writes to indexed tables).

### 3.0 Dump/backup connection
- Dumps become binary serialization of full brain state (neurons, connections, contexts, routing tables)
- Core Brain (rust) should only do file based backup/restore 
- we need separate node.js tools for importing and exporting backups to database for debugging and analysis applications
- Reference: https://claude.ai/share/e319c25b-5323-4313-adf2-d1720e068b16

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
- Sensory neurons partitioned by channel/dimension hash
- Pattern neurons live on same column as parent
- Dynamic rebalancing deferred to MPI phase (see [future-work.md](future-work.md))

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

---

## Key Design Decisions (To Refine)

1. **Neuron ownership granularity** — how many neurons per column? Fixed partition or dynamic?
2. **Cross-column connections** — neurons in different columns can have connections. How are these resolved? Proxy neurons? Message passing?
3. **Synchronization model** — lock-step frame processing across all columns, or async with eventual consistency?
4. **Memory/context scope** — is the active neuron window per-column, per-region, or global?
5. **Pattern creation across boundaries** — what happens when a pattern's parent and context neurons are on different columns?
6. **Channel assignment** — one channel per column, or all channels visible to all columns?
7. **Serialization format** — what binary format for dumps? msgpack, protobuf, custom?

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
Database                      →     Stays in JS (persistence boundary)
```

---

## Timeline Summary

| Phase | Scope | Estimate |
|-------|-------|----------|
| 1 | Unify per-neuron processing (JS refactor) | ~1.5 weeks |
| 2 | Column class design (deferred to Rust — Phase 5) | — |
| 3 | Clean up persistence (dumps/database) | ~1 week |
| 4 | Single-threaded Rust core + Node.js/npm | ~3 weeks |
| 5 | Multi-threaded Rust core + column classes | ~1 week |
| 6 | Scale stock processing | ~1 week |

**Total: ~8 weeks**

See [future-work.md](future-work.md) for Python bindings, MPI distribution, text/vision/audio channels, robotics, and other longer-term plans.

---

## Success Criteria

- **Phase 2**: Column class design documented. Implementation deferred to Phase 5 in Rust.
- **Phase 3**: Dumps are the primary backup/restore mechanism. Database is analysis-only. Serialization format is portable and versioned, ready for Rust core to own.
- **Phase 4**: Single-threaded Rust core handles frame processing. Rust unit tests pass. Dump cross-compatibility verified (JS↔Rust). JS tests pass through N-API. Published to npm. Results identical to JS implementation.
- **Phase 5**: Multi-threaded Rust core with region/column classes. Measurable speedup over single-threaded. Thread count configurable. Neurons partitioned across columns.
- **Phase 6**: Stock processing scales with parallelism. Benchmarked against JS baseline.