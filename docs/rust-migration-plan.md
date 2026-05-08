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

**Verification cadence:** Re-run after each Phase 1 implementation step. Every step must match the baseline exactly.

**Backup/restore-touching:** Verify with the [Backup → MySQL → Backup round-trip](../README.md#backup--mysql--backup-round-trip) That test asserts a rehydrated brain reproduces the same continuation result, which is exactly what changes if save/restore drifts.

Without these baselines, drift is invisible until it's already a divergence.

---

## Phase 1 — Single-Threaded Rust Core + Node.js Bindings (~2 weeks)

Rewrite the core brain computation in Rust as a single-threaded library, living in `brain-rust/`. The existing JS version remains in `brain/` until the Rust core passes all baselines and replaces it. Publish as an npm package with N-API bindings so the existing Node.js apps can call into it. Extra week budgeted for Rust learning curve (ownership/borrow checker will fight neuron graph patterns).

### 1.1 Cargo workspace setup
- Create `brain-rust/` with Cargo workspace: `brain-core` library crate + `brain-napi` binding crate
- N-API bindings via `napi-rs` for Node.js interop
- CI: `cargo build`, `cargo test`, `cargo clippy` on each commit

### 1.2 Core data structures
Port all foundational structs with ID-based references (no Rc/Arc in single-threaded phase):

#### 1.2.1 Context
- `entries: HashMap<NeuronId, HashMap<Distance, Strength>>` — nested map mirroring JS Context
- Methods: `add_neuron`, `strengthen_neuron`, `weaken_neuron`, `remove`, `has_key`, `get_entries` (flattened)
- `match_observed(observed, offset, merge_threshold, exclude_ids)` → `MatchResult { score, common, missing, novel }`
- Match scoring: full credit for exact distance, partial for delta, negative for missing

#### 1.2.2 Neuron
- `connections: HashMap<Distance, HashMap<NeuronId, ConnectionData { strength, reward }>>` — prediction map
- `routing_table: HashMap<PatternId, RoutingEntry { context: Context, activation_strength, last_activation_frame }>` — child patterns
- `context_index: HashMap<NeuronId, HashMap<Distance, HashSet<PatternId>>>` — inverted index for pattern lookup
- `context_refs: HashMap<ParentId, HashSet<Distance>>` — bidirectional parent tracking
- `error_stats: HashMap<Age, WelfordState { n, mean, m2 }>` — online variance per age
- Core methods:
  - `process_frame()` → `ProcessFrameResult { matches, correction_activations, context_ref_updates, votes }`
  - `learn_connections()` — upsert observed, weaken missing (events only)
  - `recognize_patterns()` — inverted-index-accelerated matching, context refinement
  - `correct_errors()` — install error-correction patterns
  - `generate_votes()` — per-age voting with suppression for activated ages
  - `add_pattern()`, `remove_child()`, `strengthen_child_activation()`
  - `serialize()` / deserialization helpers for backup
- Lazy decay: `effective_strength = strength - (frame - last_activation_frame) * rate`, clamped to 0

#### 1.2.3 Quantizer
- Dimension registry: `HashMap<DimId, DimensionConfig>` with mode enum `{ Passthrough, Static, Dynamic }`
- Methods: `register_dimension`, `observe`, `quantize`, `dequantize`
- Dynamic warmup: buffer samples, compute quantile boundaries post-warmup
- 1-indexed buckets `[1..=resolution]`

#### 1.2.4 Diagnostics
- Accuracy stats, reward stats, MAPE, misprediction log
- Methods: `track_inference_performance`, `track_continuous_error`, `get_stats`, `reset`

### 1.3 Thalamus (neuron registry + routing)
- Neuron metadata tables: `base_neurons`, `neuron_parents`, `neuron_levels`, `neuron_death_frame`
- Channel/dimension registries with bidirectional name↔id maps
- ID allocators: `next_channel_id`, `next_dimension_id`, `next_neuron_id`
- `neurons_by_value: HashMap<ValueKey, NeuronId>` — coordinate→neuron lookup
- Death ledger: `HashMap<FrameNumber, HashSet<NeuronId>>` — scheduled cleanup
- Channel actions: `HashMap<ChannelId, HashSet<NeuronId>>` + flat `action_ids: HashSet`
- Key methods:
  - `get_neuron_id_for_point()` — allocate or lookup sensory neuron
  - `allocate_pattern_neuron()` — build spec with resolved connections
  - `register_channel_spec()` — allocate IDs, register quantizer dims, pre-create action neurons
  - `process_level()` → `get_level_tasks`, `get_level_corrections`, `dispatch_frame`, collect results
  - `apply_level_results()` — Op-4/Op-5 flush
  - `delete_patterns()` — cascade deletion via pulse ops
  - `reap_dead_neurons()` — evict scheduled deaths
  - `materialize_and_reset_neurons()` — flatten lazy decay for save/load boundaries
  - `get_snapshot()` / `restore_snapshot()`

### 1.4 Column (neuron partition)
- `neurons: HashMap<NeuronId, Neuron>` — sole neuron storage
- Methods:
  - `process_level()` — calls `neuron.process_frame()` on each task
  - `delete_neurons()` — cascade delete ops (DeleteNeuron, RemovePattern, PurgeContextNeuron, RemoveContextRef)
  - `create_neurons()` — construct neurons from specs
  - `update_context_refs()` — Op-5 batch updates
  - `get_snapshot()` / `restore_snapshot()` / `load_neuron()`
  - `collect_death_frames()` — rebuild death ledger from routing tables
  - `materialize_and_reset_neurons()`

### 1.5 Region (column partition + routing)
- `columns: Vec<Column>` — deterministic routing via `neuron_id % C`
- Methods mirror Column but with bucketing: `process_level`, `create_neurons`, `delete_neurons`, `update_context_refs`
- Snapshot aggregation across columns

### 1.6 Memory (temporal sliding window)
- `neuron_states: HashMap<NeuronId, HashMap<FrameNumber, NeuronState>>`
- `age_index: HashMap<FrameNumber, HashSet<NeuronId>>`
- `level_index: HashMap<Level, HashMap<FrameNumber, HashSet<NeuronId>>>`
- `inferred_neurons: Vec<InferredNeuron>`
- Methods: `age()`, `activate_neuron()`, `activate_pattern()`, `get_neuron_ids_at_age()`, `get_level_ages()`, `get_level_neurons()`, `reset()`

### 1.7 Brain (orchestrator)
- Owns Thalamus, Memory, Diagnostics, Backup
- `process_frame(inputs, rewards)` → `{ inferences, frame_info }`
  - Op-1: Create sensory neurons
  - Op-2: Cleanup dead patterns (cascade)
  - Op-3: Dispatch processFrame to regions/columns (per-level)
  - Op-4: Batch-create error-correction pattern neurons
  - Op-5: Batch contextRef updates
- `build_frame()` — quantize inputs, merge inferred actions
- `age_context()` — slide temporal window, deactivate aged-out
- `process_levels()` — hierarchical level-by-level processing
- `infer_neurons()` — voting consensus (events by probability, actions by reward)
- `cleanup_dead_patterns()` — reap + cascade delete
- `register_channel_spec()`, `reset_context()`, `reset_brain()`

### 1.8 Backup (CSV serialization)
- Same CSV format as JS version for cross-compatibility:
  - `channels.csv`, `dimensions.csv`, `neurons.csv`, `base_neurons.csv`
  - `connections.csv`, `patterns.csv`, `contexts.csv`, `neuron_error_stats.csv`
- Methods: `save(job_dir, snapshot)`, `load_latest(job_dir)`
- Materialization pre-save (flatten lazy decay)
- Timestamped folders, prune old backups (keep 10)

### 1.9 N-API bindings (`brain-napi` crate)
- Expose Brain as a JS class via `napi-rs`:
  - `constructor(config)` — contextLength, errorMode, thresholds, etc.
  - `processFrame(inputs, rewards)` — returns inferences + frame info
  - `registerChannelSpec(spec)` — channel registration
  - `save(jobDir)` / `load(jobDir)` — backup/restore
  - `resetContext()` / `resetBrain()` — state management
  - `getFrameSummary()` / `getEpisodeSummary()` — diagnostics
- JS↔Rust type conversion: Maps become HashMaps, nested objects become structs
- npm package: `brain-rust/npm/` with prebuilt binaries (Windows, Linux, macOS)

### 1.10 Testing
#### 1.10.1 Rust unit tests
- Per-struct unit tests mirroring JS behavior:
  - Context: match scoring, add/strengthen/weaken
  - Neuron: learn connections, recognize patterns, vote generation, lazy decay
  - Quantizer: passthrough/static/dynamic modes, warmup, boundary computation
  - Memory: age/eviction, level indexing
  - Thalamus: routing, death ledger, channel registration
- Frame-level verification: given same inputs, Rust produces same results as JS

#### 1.10.2 Cross-compatibility testing
- Rust loads JS-produced backup dumps → produces identical continuation
- JS loads Rust-produced backup dumps → produces identical continuation
- Watch for: f64 rounding, HashMap iteration order (sort before comparison), ID sequences

#### 1.10.3 Baseline verification
- All four demos must match JS baseline exactly (see Behavior Baseline section above)
- Run after each sub-step to catch drift early

---

## Phase 2 — Multi-Threaded Rust Core + Column Classes (~1 week)

Add threading within the Rust core. Introduce the region/column abstractions directly in Rust.

### 2.1 Implement `Column` in Rust
- Owns a partition of neurons (arena-allocated)
- Calls `neuron.process_frame()` on each owned neuron
- No locks needed — single-owner per thread

### 2.2 Implement `Region` with thread pool
- Spawns N worker threads, each running a `Column`
- Uses channels (crossbeam) or shared memory for intra-column communication
- Barrier synchronization at frame boundaries
- Vote aggregation across columns

### 2.3 Neuron metadata storage for multi-threaded
- Each Region holds the metadata lookup tables (channel, type, dimension, value, level, parentId) for its owned neurons (dimension/value are level-0 only)
- Columns within the same Region read from the parent region's tables via shared memory — no copies needed
- All neuron metadata is immutable after creation, so no synchronization is required for reads
- Lookup interface remains the same as single-threaded (`get_channel(neuron_id)`, `get_type(neuron_id)`, etc.)

### 2.4 Neuron-to-thread ownership map
- Maintain a 2-way map: `neuronId ↔ threadId/columnId`
- When a neuron is created, Brain decides which thread/column owns it
- This map is used for routing: when a neuron references a foreign ID, the system knows which thread to query

### 2.5 Neuron distribution on load, collection on save
- When loading from a dump, neurons are distributed to the thread/column pool based on partitioning strategy
- When saving, Brain collects neuron state from each thread/column and serializes centrally

### 2.6 Neuron partitioning strategy
- No special-casing for sensory vs pattern, and no parent-co-location — every neuron's owner is purely a function of its id
- No dynamic rebalancing planned; revisit only if profiling surfaces a hot path

---

## Phase 3 — Scale Stock Processing (~1 week)

With multi-threaded Rust core in place, focus on scaling the stock trading workload as the primary benchmark.

### 3.1 Multi-stock parallel processing
- Multiple stock channels processed in parallel across columns
- Benchmark: throughput vs single-threaded JS baseline

### 3.2 Performance tuning
- Inverted index for pattern recognition already implemented in JS and ported to Rust (Phase 1) — validate selectivity metrics at scale
- Profile hot paths (vote aggregation, connection updates, index maintenance)
- Optimize memory layout for cache locality (arena allocation, struct-of-arrays where beneficial)
- Tune thread count and neuron partitioning for stock workloads

### 3.3 Add pre-training demo
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
| 1 | Single-threaded Rust core + Node.js/npm | ~2 weeks |
| 2 | Multi-threaded Rust core + column classes | ~1 week  |
| 3 | Scale stock processing | ~1 week  |

**Total: ~4 weeks**

See [future-work.md](future-work.md) for Python bindings, MPI distribution, text/vision/audio channels, robotics, and other longer-term plans.