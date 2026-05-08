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

## Behavior baseline

**Baseline = the four documented demos in [README](../README.md), run at R=1, C=1.** They cover synthetic cycles, real stock data, sequence memorization, and text learning — together they exercise pattern creation, error correction, deletion cascades, and convergence dynamics.

| # | Demo | Command | Headline numbers to verify |
|---|------|---------|----------------------------|
| 1 | [Demo 1 — Single-channel synthetic cycle](../README.md#demo-1-single-channel-synthetic-cycle) | `node apps/stocks/jobs/synthetic-extended-test.js --error-mode static --error-threshold 0.3 --merge-threshold 0.9` | Overall Optimal Rate (e.g. `233/240 = 97.1%`) |
| 2 | [Demo 3 — Stock trading, 1 episode](../README.md#demo-3-stock-trading) | `node apps/stocks/jobs/test.js` | Episode 1 net profit, total trades, base-level accuracy |
| 3 | [Demo 6 — Sequence memorization, **1 episode only**](../README.md#demo-6-stock-sequence-memorization) | `node apps/stocks/jobs/test.js --no-summary --episodes 1 --symbols KGC,GLD,SPY --context-length 3 --forget-rate 0.001 --error-mode static --error-threshold 0.3` | Episode 1 net profit, total trades, base-level accuracy |
| 4 | [Demo 7 — Text sequence learning](../README.md#demo-7-text-sequence-learning) | `node apps/text/jobs/test.js --file abramov.txt --error-mode static --error-threshold 0.3 --context-length 20 --merge-threshold 0.9 --forget-rate 0.001 --no-summary` | Per-episode accuracy across all 5 episodes |

**Verification cadence:** Re-run after each Phase 4 implementation step. Every step must match the baseline exactly.

**Backup/restore-touching:** Verify with the [Backup → MySQL → Backup round-trip](../README.md#backup--mysql--backup-round-trip) That test asserts a rehydrated brain reproduces the same continuation result, which is exactly what changes if save/restore drifts.

Without these baselines, drift is invisible until it's already a divergence.

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
Backup (CSV files)            →     Owned by Rust core (mirrors DB schema)
apps/db (MySQL utilities)     →     Stays in JS, outside brain core (analysis only)
```

---

## Timeline Summary

| Phase | Scope | Estimate |
|-------|-------|----------|
| 4 | Single-threaded Rust core + Node.js/npm | ~2 weeks |
| 5 | Multi-threaded Rust core + column classes | ~1 week  |
| 6 | Scale stock processing | ~1 week  |

**Total: ~4 weeks**

See [future-work.md](future-work.md) for Python bindings, MPI distribution, text/vision/audio channels, robotics, and other longer-term plans.