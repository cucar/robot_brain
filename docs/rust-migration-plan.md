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

## Phase 1 — Rayon Column Parallelism (single region)

Parallelize column processing within a Region using Rayon's work-stealing thread pool.
Single region assumed (R=1) — the Brain/Thalamus/Memory orchestration stays single-threaded.
MPI multi-region is deferred to a later phase.

### Rayon parallelism in Region

The only file that changes meaningfully is `region.rs`. Every method that loops over `self.columns` sequentially switches to Rayon's `par_iter_mut()` (or `par_iter()` for read-only).

No locks, barriers, or channels needed — each column gets exclusive `&mut` access via Rayon's parallel iterator, and all shared inputs are immutable references.

### 1.1 Add Rayon dependency

**File: `brain-core/Cargo.toml`**

```toml
[dependencies]
rayon = "1"
```

### 1.2 Parallelize `Region.process_level()` — Op-3 hot path

**File: `brain-core/src/region.rs`**

This is the critical change. Currently:
```rust
for (col_idx, task_indices) in indices_by_column.iter().enumerate() {
    let col_results = self.columns[col_idx].process_level(...);
    results.extend(col_results);
}
```

Changes to:
```rust
// pre-build per-column task lists before entering par_iter
let column_tasks: Vec<Vec<_>> = indices_by_column.iter()
    .map(|task_indices| task_indices.iter().map(|&i| tasks[i].clone()).collect())
    .collect();

// parallel dispatch — Rayon gives each column exclusive &mut access
// par_iter_mut().zip().map().collect() preserves column-index order
let results: Vec<Vec<ColumnProcessResult>> = self.columns.par_iter_mut()
    .zip(column_tasks.into_par_iter())
    .map(|(col, col_tasks)| {
        if col_tasks.is_empty() { return Vec::new(); }
        col.process_level(&col_tasks, memory_depth, level_context,
            new_error_pattern_ids, new_active_neurons, frame_number)
    })
    .collect();

results.into_iter().flatten().collect()
```

**Why this is safe:**
- Each column exclusively owns its neurons — `&mut Column` gives sole access
- Shared inputs are all immutable references that implement `Sync`:
  - `level_context: Option<&Context>` — read-only snapshot
  - `new_error_pattern_ids: &FxHashSet<NeuronId>` — read-only set
  - `new_active_neurons: &[ActiveNeuron]` — read-only slice
  - `memory_depth: u32`, `frame_number: FrameNumber` — Copy types
- Result order is deterministic: `par_iter_mut().zip().map().collect()` preserves index order in Rayon

### 1.3 Parallelize other Region methods

**File: `brain-core/src/region.rs`**

Same pattern for all column-iterating methods. Lower priority than 1.2 since these run once per frame or less:

| Method | Op | Pattern |
|--------|-----|---------|
| `create_neurons()` | Op-1/Op-4 | `par_iter_mut().zip(by_column).for_each()` |
| `update_context_refs()` | Op-5 | `par_iter_mut().zip(by_column).for_each()` |
| `delete_neurons()` | Op-2 | `par_iter_mut().zip(by_column).map().collect()` then merge results |
| `get_snapshot()` | save | `par_iter().flat_map_iter().collect()` |
| `materialize_and_reset_neurons()` | reset | `par_iter_mut().flat_map_iter().collect()` |
| `collect_death_frames()` | restore | `par_iter().flat_map_iter().collect()` |
| `restore_snapshot()` | load | `par_iter_mut().zip(specs_by_column).for_each()` |
| `clear()` | reset | `par_iter_mut().for_each()` |
| `update_action_sets()` | channel reg | `par_iter_mut().for_each()` — needs cloned refs |

### 1.4 Verify baseline demos at C=1

Run all four baseline demos with C=1 to confirm Rayon introduces no behavioral change. At C=1, Rayon's `par_iter` over a single-element Vec runs inline (no thread overhead). Output must be **identical** to the pre-Rayon commit.

| # | Demo | Key metric |
|---|------|------------|
| 1 | Synthetic cycle | Overall Optimal Rate |
| 2 | Stock trading (1 episode) | Net profit, total trades, base accuracy |
| 3 | Sequence memorization (1 episode) | Net profit, total trades, base accuracy |
| 4 | Text sequence learning | Per-episode accuracy across 5 episodes |

### 1.5 Test with C>1

Run the same demos with C=2 and C=4. The results won't be identical to C=1 (neurons route to different columns, affecting discovery order), but they should converge to similar accuracy levels and not crash or deadlock.

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
Current (Sequential Rust, R=1 C=1)     Phase 1 Target (Rayon, R=1 C>1)
─────────────────────────────────────   ─────────────────────────────────
Brain                             →     Brain (unchanged, single-thread orchestrator)
Thalamus                          →     Thalamus (unchanged, centralized metadata)
  └─ Region (sequential columns)  →       └─ Region (Rayon par_iter over columns)
       └─ Column (sequential)     →            └─ Column (one per Rayon work unit)
            └─ Neurons            →                 └─ Neurons (column-owned, exclusive)
Memory                            →     Memory (unchanged, centralized)
Quantizer                         →     Quantizer (unchanged, read-only during dispatch)
Backup                            →     Backup (unchanged)
```

Phase 2+ (MPI, future):
- Region becomes an MPI rank, each on a separate process
- Thalamus metadata may need to be replicated or sharded per region
- Memory may need per-region partitioning with cross-region aggregation

---

## Timeline Summary

| Phase | Scope | Estimate |
|-------|-------|----------|
| 1 | Multi-threaded Rust core + column classes | ~1 week  |
| 2 | Scale stock processing | ~1 week  |

**Total: ~2 weeks**

See [future-work.md](future-work.md) for Python bindings, MPI distribution, text/vision/audio channels, robotics, and other longer-term plans.
