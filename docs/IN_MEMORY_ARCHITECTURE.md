# In-Memory Brain Architecture

## Overview

This document describes the pure JavaScript in-memory implementation that replaces MySQL MEMORY tables for 100-1000x performance improvement.

## Performance Goals

- **Target**: 1ms per frame (currently 1000ms with MySQL)
- **Approach**: Replace SQL queries with indexed JavaScript data structures
- **Key Insight**: Most queries are O(N²) self-joins in SQL, but O(N) or O(1) with proper indexes in JavaScript

## Data Structures

### 1. ConnectionStore
**Purpose**: Store neuron connections with temporal distances

**Indexes**:
- `byId`: Map<connection_id, Connection> - O(1) lookup by ID
- `byFromDistance`: Map<from_neuron_id, Map<distance, Set<connection_id>>> - O(1) lookup by source + distance
- `byTo`: Map<to_neuron_id, Set<connection_id>> - O(1) lookup by target

**Critical Query**: "Get all connections from neuron X at distance D"
- **SQL**: `JOIN connections c ON c.from_neuron_id = f.neuron_id WHERE c.distance = ?` - O(N) scan
- **JS**: `connectionStore.getByFromDistance(neuronId, distance)` - O(1) hash lookup

**Memory**: ~100 bytes per connection × 100K connections = 10MB

### 2. ActiveNeuronStore
**Purpose**: Track neurons active in current sliding window

**Indexes**:
- `byKey`: Map<"neuronId:level:age", ActiveNeuron> - O(1) existence check
- `byLevelAge`: Map<level, Map<age, Set<neuron_id>>> - O(1) batch queries

**Critical Query**: "Get all neurons at level L, age A"
- **SQL**: `SELECT * FROM active_neurons WHERE level = ? AND age = ?` - O(N) scan
- **JS**: `activeNeuronStore.getByLevelAge(level, age)` - O(1) hash lookup

**Memory**: ~50 bytes per entry × 1K active = 50KB

### 3. PatternStore
**Purpose**: Store pattern definitions (pattern neuron → connections)

**Indexes**:
- `byKey`: Map<"patternId:connectionId", PatternEntry> - O(1) lookup
- `byPattern`: Map<pattern_neuron_id, Map<connection_id, strength>> - O(1) get all connections for pattern
- `byConnection`: Map<connection_id, Set<pattern_neuron_id>> - O(1) find patterns containing connection

**Critical Query**: "Find patterns containing connection C"
- **SQL**: `SELECT pattern_neuron_id FROM patterns WHERE connection_id = ?` - O(N) scan
- **JS**: `patternStore.getByConnection(connectionId)` - O(1) hash lookup

**Memory**: ~80 bytes per entry × 10K patterns × 20 connections = 16MB

### 4. PatternPeakStore
**Purpose**: Bidirectional mapping between patterns and their peak neurons

**Indexes**:
- `patternToPeak`: Map<pattern_neuron_id, peak_neuron_id> - O(1) forward lookup
- `peakToPatterns`: Map<peak_neuron_id, Set<pattern_neuron_id>> - O(1) reverse lookup

**Critical Query**: "Get all patterns owned by peak P"
- **SQL**: `SELECT pattern_neuron_id FROM pattern_peaks WHERE peak_neuron_id = ?` - O(N) scan
- **JS**: `patternPeakStore.getPatterns(peakNeuronId)` - O(1) hash lookup

**Memory**: ~40 bytes per mapping × 10K patterns = 400KB

### 5. NeuronStore
**Purpose**: Store neuron metadata and coordinates

**Structure**:
```javascript
{
  neurons: Map<neuron_id, Neuron>,
  coordinates: Map<neuron_id, Map<dimension_id, value>>,
  byDimension: Map<dimension_id, Map<value, Set<neuron_id>>>  // For finding neurons by coordinate
}
```

**Memory**: ~200 bytes per neuron × 100K neurons = 20MB

### 6. Scratch Tables (cleared each frame)
- `observedPatterns`: Map<peak_neuron_id, Set<connection_id>>
- `matchedPatterns`: Map<peak_neuron_id, Set<pattern_neuron_id>>
- `connectionInference`: Map<level, Map<connection_id, {to_neuron_id, strength}>>
- `connectionInferredNeurons`: Map<"neuronId:level:age", {strength}>
- `patternInferredNeurons`: Map<"neuronId:level:age", {strength}>
- `inferredNeurons`: Map<"neuronId:level:age", {strength}>
- `activeConnections`: Similar to ActiveNeuronStore but for connections

**Memory**: ~5MB total (cleared frequently)

## Total Memory Estimate

- Persistent data: ~50MB (neurons, connections, patterns)
- Active context: ~10MB (active neurons, scratch tables)
- **Total**: ~60MB (easily fits in RAM)

## Performance Analysis

### Current MySQL Bottlenecks

1. **Peak Detection** (lines 1076-1132 in brain.js)
   - 5 CTEs with self-joins
   - O(N²) neighborhood calculation
   - **Time**: ~500ms with 1000 connections

2. **Connection Inference** (lines 690-742)
   - 4 CTEs with self-joins
   - Runs 10× per frame (once per level)
   - **Time**: ~300ms total

3. **Pattern Matching** (lines 1155-1182)
   - Complex JOIN with GROUP BY HAVING
   - **Time**: ~100ms

4. **Network Round-Trips**
   - 50+ queries per frame
   - **Time**: ~100ms overhead

**Total**: ~1000ms per frame

### JavaScript Implementation

1. **Peak Detection** (in-memory)
   ```javascript
   // Get active connections at level - O(1)
   const activeConns = activeConnectionStore.getByLevelAge(level, 0);
   
   // Build neighborhood map - O(N) single pass
   const neighborhoods = new Map();
   for (const ac of activeConns) {
     const conn = connectionStore.get(ac.connection_id);
     const strength = conn.strength * Math.pow(decayFactor, conn.distance);
     
     // Add to target strength
     if (!neighborhoods.has(conn.to_neuron_id)) {
       neighborhoods.set(conn.to_neuron_id, { total: 0, sources: new Set() });
     }
     const target = neighborhoods.get(conn.to_neuron_id);
     target.total += strength;
     target.sources.add(conn.from_neuron_id);
   }
   
   // Find peaks - O(N) single pass
   const peaks = [];
   for (const [toNeuron, data] of neighborhoods) {
     // Calculate avg strength of competing targets - O(M) where M = neighbors
     let competitorSum = 0, competitorCount = 0;
     for (const source of data.sources) {
       const sourceConns = connectionStore.getByFromDistance(source, distance);
       for (const sc of sourceConns) {
         if (sc.to_neuron_id !== toNeuron) {
           const neighbor = neighborhoods.get(sc.to_neuron_id);
           if (neighbor) {
             competitorSum += neighbor.total;
             competitorCount++;
           }
         }
       }
     }
     
     const avgCompetitor = competitorCount > 0 ? competitorSum / competitorCount : 0;
     if (data.total >= minPeakStrength && data.total > avgCompetitor * minPeakRatio) {
       peaks.push(toNeuron);
     }
   }
   ```
   **Time**: ~5ms (200x faster)

2. **Connection Inference** (in-memory)
   - Similar approach, single-pass algorithms
   - **Time**: ~2ms per level × 10 levels = 20ms

3. **Pattern Matching** (in-memory)
   - Direct hash lookups instead of JOINs
   - **Time**: ~5ms

4. **No Network Overhead**
   - Everything in-process
   - **Time**: 0ms

**Total**: ~30ms per frame (33x faster)

### Further Optimizations

With algorithmic simplifications:
- Top-K peak selection instead of ratio-based: ~1ms
- Limit pattern size: ~0.5ms
- Reduce levels to 3-5: ~10ms total

**Optimized Total**: ~5-10ms per frame (100-200x faster)

## Implementation Strategy

### Phase 1: Core Data Structures (1 day)
- ✅ ConnectionStore
- ✅ ActiveNeuronStore  
- ✅ PatternStore
- ✅ PatternPeakStore
- ⬜ NeuronStore
- ⬜ Scratch table stores

### Phase 2: Algorithm Conversion (2-3 days)
- ⬜ Peak detection
- ⬜ Connection inference
- ⬜ Pattern matching
- ⬜ Pattern merging
- ⬜ Reward propagation

### Phase 3: Integration (1-2 days)
- ⬜ Replace MySQL calls in brain.js
- ⬜ Keep MySQL for persistence (save/load)
- ⬜ Add serialization for checkpoints

### Phase 4: Testing & Optimization (1-2 days)
- ⬜ Verify correctness vs MySQL
- ⬜ Profile and optimize hot paths
- ⬜ Add memory monitoring

## Persistence Strategy

**Option A**: Hybrid (Recommended)
- Use JavaScript for all frame processing
- Periodically save to MySQL for persistence
- Load from MySQL on startup

**Option B**: Pure JavaScript
- Use JSON files for save/load
- Faster but less queryable

**Option C**: SQLite
- Embedded database for persistence
- Better than MySQL for single-process

## Risk Mitigation

1. **Memory Leaks**: Use WeakMaps where appropriate, monitor heap
2. **Correctness**: Unit tests comparing JS vs MySQL results
3. **Debugging**: Add extensive logging, visualization tools
4. **Rollback**: Keep MySQL code path as fallback

## Expected Results

- **Frame time**: 1000ms → 10-30ms (33-100x improvement)
- **Memory usage**: 2GB MySQL → 60MB JavaScript (33x reduction)
- **Throughput**: 1 frame/sec → 30-100 frames/sec
- **Training time**: 525 days in hours → minutes

This makes real-time learning feasible!

