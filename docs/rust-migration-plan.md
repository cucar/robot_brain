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
| **Neuron** | Connections, children, voting, learning, decay | connections, children, context, coordinates |
| **Context** | Pattern context matching & merging | entries Map<neuron, Map<distance, strength>> |
| **Channel** | I/O interface (stock, text, vision, etc.) | dimensions, actions, rewards |
| **Database** | MySQL persistence | connection, backup/restore |
| **Diagnostics** | Debug output & accuracy tracking | accuracy stats, mispredictions |

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

**Target**: all 4 operations merge into a level-by-level loop. At each level, for each neuron, a single `neuron.processFrame(input)` call handles: error correction install → recognize → learn connections → cast votes. Error correction decisions are made by Brain *before* the loop; neurons just receive the result.

**What stays global (never per-neuron)**: error correction decisions (Brain compares previous-frame consensus against actual events), consensus determination, action execution, ensuring channel actions, sensor activation, cleanup.

### Dependencies Between Operations (Within a Single Frame)

```
Brain.determineErrorCorrections() ──► pre-loop: compares previous votes vs actuals
                                      creates pattern neurons via Thalamus
                                      produces Map<parentNeuronId, {patternId, context}>
                                      │
                                      ▼
processLevels (level-by-level loop):
  for each neuron at this level:
    neuron.processFrame(input):
      1. installErrorCorrection ──► adds pattern to routing table (if Brain assigned one)
                                    suppresses recognition and voting
      2. recognizePatterns ────────► activates child patterns in routing table
                                    suppresses voting if pattern matched
      3. updateConnections ────────► independent — uses age=0 sensory neurons
      4. collectVotes ─────────────► skipped if suppressed by step 1 or 2
                                      │
                                      ▼
  post-processing (sequential):
    deliver contextRef updates to target neurons
    activate recognized patterns in level+1 of active neurons map
    register death frames for activated patterns
    store votes for next frame's error correction
```

### Active Neurons Data Structure Change

Current: `Array<Map<neuronId, state>>` indexed by age (age-first).

Target: dual-indexed structure to support both level-first iteration and age-based sliding window:

```
// Primary structure for the processing loop — level-first
activeLevels: Map<level, Map<neuronId, {
    ages: number[],                  // sorted ascending — all ages this neuron is active at
    activatedPatternId: number|null, // set during processFrame
    votes: Vote[]|null,              // saved for next frame's error correction
    votingContext: Context|null,     // saved for next frame's error correction
}>>

// Secondary structure for aging/eviction — age-first bookkeeping
ageSlots: Array<Set<{neuronId, level}>>
// ageSlots[0] = newest, ageSlots[N] = oldest
// On age(): unshift new empty set, pop oldest, update activeLevels accordingly
```

When `age()` is called: shift all age slots, evict aged-out neurons from `activeLevels` (decrement or remove their ages entry). When `activateNeuron(id, level)` is called: add to `ageSlots[0]` and upsert into `activeLevels[level]`.

### neuron.processFrame() — Input/Output Contract

The neuron receives a work packet with everything it needs and returns results. It never reaches outside itself — no Thalamus access, no Memory access, no cross-neuron mutations.

**Input (assembled by caller):**

```
NeuronFrameInput {
    ages: number[],                    // all ages this neuron is active at (sorted ascending)
    levelContext: Context,             // observed context at this neuron's level (read-only snapshot)
    sensoryNeurons: Array<Array>,      // age-indexed sensory neurons for connection learning
    rewards: Array<Map>,               // age-indexed rewards per channel
    channelActionIds: Map,             // for alternative action lookup in connection learning
    currentFrame: number,
    contextLength: number,

    // Error correction — decided by Brain before the loop, not by this neuron
    // null if Brain didn't assign an error correction to this neuron
    errorCorrection: {
        patternId: number,             // already created by Thalamus
        context: Array<{neuronId, distance}>,  // what to store in routing table
    } | null,
}
```

**Output (processed by caller in sequential post-processing):**

```
NeuronFrameOutput {
    // From error correction install — contextRef adds for the new pattern's context neurons
    errorCorrectionDeathFrame: number|null,
    errorCorrectionContextRefs: Array<{neuronId, distance}>,

    // From recognition
    matches: Array<{patternId, age, deathFrame, activate}>,
    contextRefUpdates: Array<{type: 'add'|'remove', neuronId, distance}>,

    // Votes (null if suppressed by error correction or recognition)
    votes: Vote[]|null,
    votingContext: Context|null,       // snapshot saved for next frame's error correction

    suppressed: boolean,
}
```

**neuron.processFrame() implementation:**

```
processFrame(input: NeuronFrameInput): NeuronFrameOutput {
    let suppressed = false;
    let errorCorrectionDeathFrame = null;
    let errorCorrectionContextRefs = [];

    // 1. INSTALL ERROR CORRECTION (Brain already decided and created the pattern)
    if (input.errorCorrection) {
        const { patternId, context } = input.errorCorrection;
        errorCorrectionDeathFrame = this.addPattern(patternId, context, input.currentFrame);
        // Collect contextRef additions — caller delivers to target neurons
        errorCorrectionContextRefs = context.map(c => ({ neuronId: c.neuronId, distance: c.distance }));
        suppressed = true;  // don't recognize, don't vote
    }

    // 2. RECOGNIZE PATTERNS (skip if error-corrected)
    let matches = [], contextRefUpdates = [];
    if (!suppressed) {
        const recognitionAges = input.ages.filter(a => a > 0);
        if (recognitionAges.length > 0) {
            const result = this.matchPatterns(input.levelContext, recognitionAges, input.currentFrame);
            matches = result.matches;
            contextRefUpdates = result.contextRefUpdates;
            if (matches.some(m => m.activate)) suppressed = true;
        }
    }

    // 3. UPDATE CONNECTIONS (always runs — independent of recognition/error correction)
    for (const age of input.ages) {
        if (age > 0)
            this.updateConnections(age, input.sensoryNeurons, input.rewards, input.channelActionIds);
    }

    // 4. COLLECT VOTES (skip if suppressed)
    let votes = null, votingContext = null;
    if (!suppressed) {
        votes = [];
        for (const age of input.ages)
            if (age < input.contextLength - 1)
                votes.push(...this.vote(age));
        votingContext = input.levelContext;  // snapshot for next frame
    }

    return {
        errorCorrectionDeathFrame, errorCorrectionContextRefs,
        matches, contextRefUpdates,
        votes, votingContext, suppressed,
    };
}
```

### processLevels() — The Unified Caller Loop

```
processLevels(sensoryNeurons, rewards, channelActionIds, currentFrame, contextLength):

    // ── Pre-loop: Brain decides error corrections ──
    // Brain compares previous-frame consensus votes (stored per-neuron in activeLevels)
    // against actual events (age-0 sensory neurons). For each neuron that voted wrong:
    //   1. Brain calls thalamus.addPatternNeuron(level+1, parentId, age, ...)
    //   2. Thalamus creates neuron, sets up connections, returns patternId
    //   3. Brain stores result in errorCorrections map
    //   4. Brain delivers addContextRef to context neurons (immediate, before loop)
    //   5. Brain activates the new pattern neuron in activeLevels at level+1
    //   6. Brain registers deathFrame
    //
    // Wait — steps 4-6 overlap with what processFrame returns. Two options:
    //   A) Brain does the full setup before the loop (addPattern happens in Thalamus,
    //      not in neuron.processFrame). Neuron just gets told "you're suppressed."
    //   B) Brain creates the pattern neuron but neuron.processFrame installs it
    //      (addPattern) and returns the side effects for the caller to deliver.
    //
    // Option B is cleaner: the neuron's routing table is its own state. Having Brain
    // mutate it from outside breaks encapsulation. So Brain pre-creates the pattern
    // neuron (with connections) via Thalamus, then passes {patternId, context} to the
    // neuron. The neuron calls addPattern (updating its own routing table + contextIndex)
    // and returns the cross-neuron side effects (contextRefs, deathFrame) for delivery.

    errorCorrections = brain.determineErrorCorrections(previousVotes, actualEvents)
    // Returns: Map<parentNeuronId, {patternId, context: Array<{neuronId, distance}>}>

    allVotes = []

    for level = 0 to maxActiveLevel:
        neuronsAtLevel = activeLevels.get(level)
        if !neuronsAtLevel or neuronsAtLevel.size == 0: continue

        // Build level context ONCE before iterating neurons at this level.
        // This is a snapshot — it does NOT change as neurons at this level are processed.
        // Consequence: if neuron A's pattern recognition activates a pattern at this same
        // level, neuron B (processed later) does NOT see it in the levelContext.
        // This is intentional: it makes intra-level processing order-independent,
        // which is required for parallel execution (Rayon in Rust).
        levelContext = buildLevelContext(level)

        // ── Parallel phase (future: Rayon par_iter in Rust) ──
        results = []
        for (neuronId, state) in neuronsAtLevel:
            neuron = thalamus.getNeuron(neuronId)
            input = {
                ages: state.ages,
                levelContext,                          // shared read-only snapshot
                sensoryNeurons, rewards, channelActionIds, currentFrame, contextLength,
                errorCorrection: errorCorrections.get(neuronId) ?? null,
            }
            output = neuron.processFrame(input)
            results.push({neuronId, state, output})

        // ── Sequential post-processing (cross-neuron side effects) ──
        newActivationsAtNextLevel = false

        for {neuronId, state, output} in results:

            // Error correction: deliver contextRefs and register death
            if output.errorCorrectionDeathFrame != null:
                for ref in output.errorCorrectionContextRefs:
                    thalamus.getNeuron(ref.neuronId).addContextRef(neuronId, ref.distance)
                thalamus.registerDeath(errorCorrections.get(neuronId).patternId,
                                       output.errorCorrectionDeathFrame)
                // Pattern neuron already activated in activeLevels by Brain pre-loop
                newActivationsAtNextLevel = true

            // Recognition: activate matched patterns at level+1
            for match in output.matches:
                if match.activate:
                    activateInLevelMap(match.patternId, level + 1, match.age)
                    thalamus.registerDeath(match.patternId, match.deathFrame)
                    newActivationsAtNextLevel = true

            // Deliver contextRef updates from recognition refinement
            for update in output.contextRefUpdates:
                targetNeuron = thalamus.getNeuron(update.neuronId)
                if update.type == 'add':
                    targetNeuron.addContextRef(neuronId, update.distance)
                else:
                    targetNeuron.removeContextRef(neuronId, update.distance)

            // Store votes for consensus and for next frame's error correction
            if output.votes:
                allVotes.push(...output.votes)
            state.votes = output.votes
            state.votingContext = output.votingContext

        // Early exit: if no new pattern activations at level+1 AND no neurons
        // exist above this level in activeLevels, stop the level loop.
        // Connection updates and voting for already-active higher-level neurons
        // have already been processed (they were in activeLevels from prior frames).
        if !newActivationsAtNextLevel and level >= maxActiveLevel:
            break

    return allVotes
```

### Incremental Implementation Steps

Each step is a self-contained refactor. Run ALL tests after each step. Results must be identical.

#### Step 1.1 — Restructure Memory for level-first indexing

**Why**: The unified loop iterates level-by-level, but Memory currently indexes by age. We need level-first access with per-neuron age arrays.

**Current Memory structure**:
```
activeNeurons: Array<Map<neuronId, {activatedPatternId, votes, context}>>
// indexed by age: activeNeurons[0] = age 0, activeNeurons[N] = age N
```

**Target Memory structure**:
```
// Primary: level-first for the processing loop
activeLevels: Map<level, Map<neuronId, {
    ages: number[],                  // sorted ascending
    activatedPatternId: number|null,
    votes: Vote[]|null,
    votingContext: Context|null,
}>>

// Secondary: age-first for sliding window mechanics (age/evict)
ageSlots: Array<Set<{neuronId, level}>>
```

**Implementation**:
1. Add `activeLevels` alongside existing `activeNeurons` — populate both during activation, verify they stay in sync
2. Convert all loop consumers (recognizePatterns, updateConnections, collectVotes, learnNewPatterns) to read from `activeLevels` instead of `activeNeurons`
3. Convert `age()` to maintain both structures — shift ageSlots, update ages arrays in activeLevels, remove neurons whose last age is evicted
4. Remove old `activeNeurons` once no code reads it
5. Rename `votes`/`context` fields to `votes`/`votingContext` for clarity

**Key behavioral decision**: `activatePattern()` currently inserts the pattern neuron at the same age as the parent. With level-first indexing, it inserts into `activeLevels[level+1]` with the parent's age. This makes the pattern immediately available when the loop reaches level+1.

**Verify**: all iteration patterns produce identical neuron sets, all tests pass

#### Step 1.2 — Move error correction decisions before the level loop

**Why**: Error correction is a Brain-level decision (compare consensus votes against actual events). Moving it before the loop makes neurons passive recipients — they just install the pattern they're told to install.

**Current flow**:
* `learnNewPatterns()` iterates neurons with previous-frame votes
* For each, Brain checks if the neuron's predictions were wrong
* If wrong, Brain creates a new pattern via Thalamus and calls neuron methods to set it up

**Target flow**:
* Before the level loop, Brain calls `determineErrorCorrections()`:
    - Iterates `activeLevels` looking for neurons with stored `votes` from previous frame
    - For each, compares votes against actual events (age-0 sensory neurons)
    - If error exceeds threshold: calls `thalamus.addPatternNeuron(level+1, parentId, age, sensoryNeurons, rewards, levelContext, currentFrame)`
    - Thalamus creates the pattern neuron with connections but does NOT install it in the parent's routing table — that happens inside `neuron.processFrame()`
    - Brain stores `{patternId, context}` in `errorCorrections` map keyed by `parentNeuronId`
    - Brain activates the new pattern neuron in `activeLevels[level+1]`
* During the level loop, `errorCorrections.get(neuronId)` is passed as input to `neuron.processFrame()`
* The neuron calls `this.addPattern(patternId, context, currentFrame)` — updating its own routing table
* The neuron returns `errorCorrectionContextRefs` and `errorCorrectionDeathFrame` for the caller to deliver

**Important**: `thalamus.addPatternNeuron()` must be split — currently it both creates the neuron AND installs it in the parent's routing table (`parent.addPattern(...)`). After this step, Thalamus creates the neuron and its connections, but the routing table installation is deferred to `neuron.processFrame()`.

**Verify**: pattern creation identical, error correction timing identical, all tests pass

#### Step 1.4 — Move `collectVotes` into the level loop

* Currently `collectVotes()` iterates all active neurons at ages 0..N-1 across all levels
* Restructure: within the level loop, the neuron's `processFrame()` calls `this.vote(age)` for each eligible age
* Votes are returned in `NeuronFrameOutput` and accumulated by the caller
* Suppression: `processFrame()` skips voting if `errorCorrection` was installed or if recognition activated a pattern
* Previous-frame votes and context are stored in `activeLevels` state for next frame's error correction
* Remove standalone `collectVotes()` from Brain once fully merged
* **Verify**: votes identical, inference results identical, all tests pass

### Phase 1 Success Criteria

* All existing tests pass identically
* Every neuron's per-frame work goes through a single `processFrame(input): output` call
* `processFrame()` has no side effects outside the neuron itself — all cross-neuron effects are returned as output
* Error correction decisions are made by Brain before the level loop — neurons just install pre-created patterns
* Memory is indexed level-first with per-neuron age arrays
* Brain delegates level iteration to Thalamus via `thalamus.processLevels(...)`
* Level context is a snapshot taken before iterating neurons at each level (order-independent)
* All neuron references are ID-based (from Step 1.0)
* Contexts live in parent routing tables with inverted contextIndex (from Step 1.1)

---

## Phase 2 — Introduce Column Classes (deferred to Rust — Phase 5)

Column abstractions (Column, Region) will be introduced directly in Rust rather than building them in JS only to throw them away 2–3 weeks later. The JS refactoring in Phase 1 already validates that per-neuron processing is clean enough to parallelize. The column classes are a Rust-native concern.

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
- Neuron channel, type, coordinates belong to base level only - may be ok to merge as baseNeurons?
- Thalamus tracks which region/column owns each neuron
- Neuron lookup still global (Thalamus), but mutations route through owner

### 2.5 Refactor Memory for per-column active state
- Each column has its own active neuron window
- Global Memory becomes an aggregator over per-column memories
- Inferred neurons remain global (consensus output)

### 2.6 Cleanup
- Change offset rows to work from the end and add demo for training first and then testing accuracy with hold out data
- Change brain interface to take inputs of the frame as arguments - maybe even actions?

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
- Each Region holds the metadata lookup tables (channel, type, coordinates, level, parentId) for its owned neurons
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

- **Phase 1**: All existing tests pass identically. Brain behavior unchanged. Every neuron's frame work goes through a single `processFrame()` call. Brain delegates to Thalamus, not to individual neuron loops. All neuron references are ID-based. Contexts live in parent routing tables.
- **Phase 2**: Column class design documented. Implementation deferred to Phase 5 in Rust.
- **Phase 3**: Dumps are the primary backup/restore mechanism. Database is analysis-only. Serialization format is portable and versioned, ready for Rust core to own.
- **Phase 4**: Single-threaded Rust core handles frame processing. Rust unit tests pass. Dump cross-compatibility verified (JS↔Rust). JS tests pass through N-API. Published to npm. Results identical to JS implementation.
- **Phase 5**: Multi-threaded Rust core with region/column classes. Measurable speedup over single-threaded. Thread count configurable. Neurons partitioned across columns.
- **Phase 6**: Stock processing scales with parallelism. Benchmarked against JS baseline.