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

---

## Phase 1 — Multi-Threaded Rust Core + Column Classes (~1 week)

Add threading within the Rust core. Introduce the region/column abstractions directly in Rust.

### 1.1 Implement `Column` in Rust
- Owns a partition of neurons (arena-allocated)
- Calls `neuron.process_frame()` on each owned neuron
- No locks needed — single-owner per thread

### 1.2 Implement `Region` with thread pool
- Spawns N worker threads, each running a `Column`
- Uses channels (crossbeam) or shared memory for intra-column communication
- Barrier synchronization at frame boundaries
- Vote aggregation across columns

### 1.3 Neuron metadata storage for multi-threaded
- Each Region holds the metadata lookup tables (channel, type, dimension, value, level, parentId) for its owned neurons (dimension/value are level-0 only)
- Columns within the same Region read from the parent region's tables via shared memory — no copies needed
- All neuron metadata is immutable after creation, so no synchronization is required for reads
- Lookup interface remains the same as single-threaded (`get_channel(neuron_id)`, `get_type(neuron_id)`, etc.)

### 1.4 Neuron-to-thread ownership map
- Maintain a 2-way map: `neuronId ↔ threadId/columnId`
- When a neuron is created, Brain decides which thread/column owns it
- This map is used for routing: when a neuron references a foreign ID, the system knows which thread to query

### 1.5 Neuron distribution on load, collection on save
- When loading from a dump, neurons are distributed to the thread/column pool based on partitioning strategy
- When saving, Brain collects neuron state from each thread/column and serializes centrally

### 1.6 Neuron partitioning strategy
- No special-casing for sensory vs pattern, and no parent-co-location — every neuron's owner is purely a function of its id
- No dynamic rebalancing planned; revisit only if profiling surfaces a hot path

---

## Phase 2 — Scale Stock Processing (~1 week)

With multi-threaded Rust core in place, focus on scaling the stock trading workload as the primary benchmark.

### 2.1 Multi-stock parallel processing
- Multiple stock channels processed in parallel across columns
- Benchmark: throughput vs single-threaded baseline

### 2.2 Performance tuning
- Inverted index for pattern recognition already implemented — validate selectivity metrics at scale
- Profile hot paths (vote aggregation, connection updates, index maintenance)
- Optimize memory layout for cache locality (arena allocation, struct-of-arrays where beneficial)
- Tune thread count and neuron partitioning for stock workloads

### 2.3 Add pre-training demo
- Change offset rows to work from the end and add demo for training first and then testing accuracy with hold out data

---

## Mapping: Current Architecture → Target Architecture

```
Current (Single-Threaded Rust)          Target (Multi-Threaded Rust + MPI)
─────────────────────────────────────   ─────────────────────────────────
Brain (Rust via N-API)            →     Brain (Rust via N-API)
                                          ├── Region (MPI rank 0)
                                          │     ├── Column thread 0
                                          │     │     ├── Neuron pool
                                          │     │     ├── Local memory
                                          │     │     └── Local pattern recognition
                                          │     ├── Column thread 1
                                          │     └── Vote aggregator
                                          ├── Region (MPI rank 1)
                                          └── Global consensus (MPI)

Thalamus                          →     Split: per-column registry + global lookup
Memory                            →     Split: per-column window + region aggregator
Neuron                            →     Rust struct (arena-allocated, column-owned)
Context                           →     Rust struct (inline with neuron)
Backup (CSV files)                →     Owned by Rust core (mirrors DB schema)
apps/db (MySQL utilities)         →     Stays in JS, outside brain core (analysis only)
```

---

## Timeline Summary

| Phase | Scope | Estimate |
|-------|-------|----------|
| 1 | Multi-threaded Rust core + column classes | ~1 week  |
| 2 | Scale stock processing | ~1 week  |

**Total: ~2 weeks**

See [future-work.md](future-work.md) for Python bindings, MPI distribution, text/vision/audio channels, robotics, and other longer-term plans.
