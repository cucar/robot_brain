# Error-Driven Learning Architecture - Implementation Plan

## Overview
This document outlines the implementation plan for refactoring the brain architecture from continuous pattern creation to error-driven learning with top-down inference.

## Core Architectural Changes

### 1. Pattern Recognition → Match Only (No Creation)
**Current Behavior:**
- `activateLevelPatterns()` detects peaks, matches patterns, merges (reinforces), and creates new patterns
- Pattern creation happens during every recognition cycle

**New Behavior:**
- Pattern recognition only matches and activates existing patterns
- No pattern creation during recognition
- No reinforcement during recognition (positive or negative)

**Changes Required:**
- Remove `mergeMatchedPatterns()` call from `activateLevelPatterns()`
- Remove `createNewPatterns()` call from `activateLevelPatterns()`
- Keep only: `getObservedPatterns()`, `matchObservedPatterns()`, and pattern neuron activation

### 2. Connection Learning → Hebbian Only (No Reinforcement on Activation)
**Current Behavior:**
- `activateNeurons()` calls `reinforceConnections()` which strengthens existing connections
- Connections are created/strengthened whenever neurons co-occur

**New Behavior:**
- Connection creation still happens (Hebbian: neurons that fire together wire together)
- Connection reinforcement (strengthening) removed from activation
- Connections only strengthened during error-driven learning

**Changes Required:**
- Remove `reinforceConnections()` call from `activateNeurons()`
- Keep connection creation logic (INSERT with strength=1)
- Remove ON DUPLICATE KEY UPDATE logic from connection creation

### 3. Error-Driven Pattern Creation
**New Phase:** `validateAndLearnFromErrors()` - runs before `inferNeurons()`

**Process:**
1. Check predictions from previous frame (age=1 in connection_inferred_neurons)
2. For each failed prediction where strength >= `minErrorPatternThreshold`:
   - Get all connections from connection_inference that predicted this neuron
   - Create a new pattern neuron at level+1
   - Store connections in pattern_past table (for learning/matching)
   - Store connections in pattern_future table (for inference unpacking)
3. Apply negative reinforcement to failed connection predictions (existing logic)

**New Hyperparameter:**
```javascript
this.minErrorPatternThreshold = 0.5; // minimum prediction strength to create pattern from error
```

**New Tables Required:**
```sql
-- Split patterns table into past and future
-- pattern_past: connections leading TO the peak (for recognition/matching)
CREATE TABLE IF NOT EXISTS pattern_past (
    pattern_neuron_id BIGINT UNSIGNED NOT NULL,
    connection_id BIGINT UNSIGNED NOT NULL,
    strength DOUBLE NOT NULL DEFAULT 1.0,
    PRIMARY KEY (pattern_neuron_id, connection_id),
    FOREIGN KEY (pattern_neuron_id) REFERENCES neurons(id) ON DELETE CASCADE,
    FOREIGN KEY (connection_id) REFERENCES connections(id) ON DELETE CASCADE,
    INDEX idx_connection_strength (connection_id, strength),
    INDEX idx_pattern_strength (pattern_neuron_id, strength),
    INDEX idx_strength (strength)
) ENGINE=MEMORY;

-- pattern_future: connections FROM the peak (for inference unpacking)
CREATE TABLE IF NOT EXISTS pattern_future (
    pattern_neuron_id BIGINT UNSIGNED NOT NULL,
    connection_id BIGINT UNSIGNED NOT NULL,
    strength DOUBLE NOT NULL DEFAULT 1.0,
    PRIMARY KEY (pattern_neuron_id, connection_id),
    FOREIGN KEY (pattern_neuron_id) REFERENCES neurons(id) ON DELETE CASCADE,
    FOREIGN KEY (connection_id) REFERENCES connections(id) ON DELETE CASCADE,
    INDEX idx_pattern_strength (pattern_neuron_id, strength),
    INDEX idx_strength (strength)
) ENGINE=MEMORY;
```

**Migration Required:**
- Rename existing `patterns` table to `pattern_past`
- Create new `pattern_future` table
- Populate `pattern_future` from existing pattern data (connections where distance > 0)

### 4. Top-Down Inference Flow
**Current Behavior:**
- `inferConnections()` - predicts for all levels at once
- `inferPatterns()` - recursive CTE cascading down from connection predictions
- Bottom-up flow: base connections → pattern cascade

**New Behavior:**
- Start from highest active level, work down to level 0
- At each level:
  1. **Connection inference** (same-level predictions)
  2. **Pattern inference** (level predicting level-1 via pattern_future unpacking)
  3. If pattern predictions exist → unpack and filter → recurse down
  4. Otherwise → recurse to connection inference of level below

**Unpacking Process:**
For pattern predictions at level N predicting level N-1:
1. Get predicted neurons at level N-1 (from pattern_future connections)
2. For each predicted neuron, get its future connections (from pattern_future if it's a pattern neuron)
3. Calculate to_neurons and their strengths (with peakTimeDecayFactor weighting)
4. Calculate average strength
5. Filter neurons above average strength
6. Write to inferred_neurons at level N-2
7. Recurse down

**New Method Structure:**
```javascript
async inferNeurons() {
    // Report accuracy from previous frame
    await this.reportPredictionsAccuracy();
    
    // Get highest active level
    const maxActiveLevel = await this.getMaxActiveLevel();
    
    // Start top-down inference from highest level
    for (let level = maxActiveLevel; level >= 0; level--) {
        await this.inferFromLevel(level);
    }
    
    // Resolve conflicts for base level (level 0)
    await this.resolveInputPredictionConflicts();
}

async inferFromLevel(level) {
    // Step 1: Connection inference at this level
    await this.inferConnectionsAtLevel(level);
    
    // Step 2: Pattern inference (this level predicting level-1)
    if (level > 0) {
        const patternPredictions = await this.inferPatternsFromLevel(level);
        
        if (patternPredictions.length > 0) {
            // Unpack pattern predictions recursively
            await this.unpackPatternPredictions(patternPredictions, level - 1);
        }
    }
}

async unpackPatternPredictions(predictions, targetLevel) {
    // Get future connections from predicted neurons
    const futureConnections = await this.getFutureConnections(predictions);
    
    // Calculate to_neurons with weighted strengths
    const inferredNeurons = await this.getInferredNeuronsFromConnections(futureConnections);
    
    // Filter above average
    const avgStrength = this.calculateAverageStrength(inferredNeurons);
    const filteredNeurons = inferredNeurons.filter(n => n.strength > avgStrength);
    
    // Write to pattern_inferred_neurons
    await this.writePatternInferredNeurons(filteredNeurons, targetLevel);
    
    // Recurse down if target level > 0
    if (targetLevel > 0 && filteredNeurons.length > 0) {
        await this.unpackPatternPredictions(filteredNeurons, targetLevel - 1);
    }
}
```

## Implementation Steps

### Phase 1: Database Schema Changes
1. **Create new tables:**
   - `pattern_past` (copy of current `patterns` structure)
   - `pattern_future` (same structure as `pattern_past`)

2. **Migration script:**
   - Rename `patterns` → `pattern_past`
   - Create `pattern_future`
   - Populate `pattern_future` from `pattern_past` (initially same data)
   - Update all references to `patterns` table → `pattern_past`

3. **Update forget cycle:**
   - Apply forget rate to both `pattern_past` and `pattern_future`
   - Delete patterns with strength <= minConnectionStrength from both tables

### Phase 2: Remove Pattern Creation from Recognition
1. **Modify `activateLevelPatterns()`:**
   - Remove `mergeMatchedPatterns()` call
   - Remove `createNewPatterns()` call
   - Keep only matching and activation logic

2. **Update `matchObservedPatterns()`:**
   - Change references from `patterns` → `pattern_past`
   - No other logic changes needed

### Phase 3: Remove Connection Reinforcement from Activation
1. **Modify `activateNeurons()`:**
   - Remove `reinforceConnections()` call
   - Keep `insertActiveNeurons()` call
   - Keep `activateConnections()` call

2. **Modify `reinforceConnections()` (if still used elsewhere):**
   - Change INSERT ... ON DUPLICATE KEY UPDATE to just INSERT
   - Remove strength increment logic
   - Or delete method entirely if only used in activation

### Phase 4: Implement Error-Driven Learning
1. **Add hyperparameter:**
   ```javascript
   this.minErrorPatternThreshold = 0.5;
   ```

2. **Create `validateAndLearnFromErrors()` method:**
   - Check failed predictions from connection_inferred_neurons (age=1)
   - Filter by strength >= minErrorPatternThreshold
   - For each failed prediction:
     - Get connections from connection_inference
     - Create pattern neuron at level+1
     - Insert into pattern_past (connections leading to peak)
     - Insert into pattern_future (connections from peak)
     - Create pattern_peaks mapping

3. **Update `processFrame()` flow:**
   ```javascript
   async processFrame(frame, globalReward = 1.0) {
       await this.applyRewards(globalReward);
       await this.ageNeurons();
       await this.executeOutputs();
       await this.recognizeNeurons(frame);
       
       // NEW: Error-driven learning before inference
       await this.validateAndLearnFromErrors();
       
       await this.inferNeurons();
       await this.runForgetCycle();
   }
   ```

### Phase 5: Implement Top-Down Inference
1. **Create helper method:**
   ```javascript
   async getMaxActiveLevel() {
       const [result] = await this.conn.query(
           'SELECT MAX(level) as max_level FROM active_neurons WHERE age = 0'
       );
       return result[0].max_level || 0;
   }
   ```

2. **Refactor `inferConnections()` → `inferConnectionsAtLevel(level)`:**
   - Add level parameter
   - Filter active_neurons by level
   - Keep same peak detection logic
   - Write to connection_inferred_neurons

3. **Create `inferPatternsFromLevel(level)` method:**
   - Get active neurons at this level
   - Check if they are pattern neurons (in pattern_peaks)
   - Get their pattern_future connections
   - Return predicted neurons at level-1

4. **Create `unpackPatternPredictions()` method:**
   - Recursive unpacking as described above
   - Use pattern_future table for connections
   - Apply peakTimeDecayFactor weighting
   - Filter above average strength

5. **Refactor `inferNeurons()` method:**
   - Remove current `inferConnections()` and `inferPatterns()` calls
   - Implement top-down loop as shown above
   - Keep `mergeHigherLevelPredictions()` for level > 0
   - Keep `resolveInputPredictionConflicts()` for level 0

### Phase 6: Update Negative Reinforcement
1. **Connection negative reinforcement:**
   - Keep existing `negativeReinforceConnections()` logic
   - Uses connection_inference table (already correct)

2. **Pattern negative reinforcement:**
   - Similar to connection negative reinforcement
   - Check pattern_inferred_neurons (age=1) against active_neurons
   - Weaken pattern_past connections that failed
   - Weaken pattern_future connections that failed

## Testing Strategy

### Unit Tests
1. **Test pattern table split:**
   - Verify pattern_past and pattern_future are populated correctly
   - Verify forget cycle works on both tables

2. **Test recognition without creation:**
   - Verify patterns are matched but not created during recognition
   - Verify no reinforcement during recognition

3. **Test error-driven learning:**
   - Create scenario with confident wrong prediction
   - Verify pattern created at level+1
   - Verify pattern_past and pattern_future populated correctly

4. **Test top-down inference:**
   - Create multi-level active neurons
   - Verify inference starts from highest level
   - Verify unpacking works correctly
   - Verify filtering above average works

### Integration Tests
1. **Test full learning cycle:**
   - Run multiple frames with errors
   - Verify patterns created only on errors
   - Verify patterns used in subsequent predictions

2. **Test hierarchical learning:**
   - Create scenarios requiring multi-level patterns
   - Verify patterns created at appropriate levels
   - Verify top-down inference uses hierarchical patterns

## Migration Path

### Step 1: Schema Migration (Non-Breaking)
- Run migration script to create pattern_past and pattern_future
- Keep old patterns table temporarily
- Update code to use pattern_past for reads

### Step 2: Code Changes (Breaking)
- Implement all Phase 2-5 changes
- Test thoroughly with existing data

### Step 3: Cleanup
- Drop old patterns table
- Remove deprecated methods

## Rollback Plan
- Keep backup of patterns table
- Keep old code in git branch
- If issues found, restore patterns table and revert code

## Performance Considerations
1. **Pattern table split:** Minimal impact, same total rows
2. **Error-driven learning:** Reduces pattern creation overhead during recognition
3. **Top-down inference:** May be slower than bulk processing, but more accurate
4. **Unpacking recursion:** Monitor stack depth, may need iterative approach for deep hierarchies

## Open Questions
1. Should pattern_future store ALL connections or only distance > 0?
2. How to handle pattern_past vs pattern_future strength divergence over time?
3. Should we limit pattern creation to one per failed prediction or allow multiple?
4. What's the optimal minErrorPatternThreshold value?

## Detailed Implementation Notes

### Error-Driven Pattern Creation Logic

**Pseudocode for `validateAndLearnFromErrors()`:**
```javascript
async validateAndLearnFromErrors() {
    // Get failed predictions with high confidence
    const [failures] = await this.conn.query(`
        SELECT cin.neuron_id, cin.level, cin.strength
        FROM connection_inferred_neurons cin
        WHERE cin.age = 1
        AND cin.strength >= ?
        AND NOT EXISTS (
            SELECT 1 FROM active_neurons an
            WHERE an.neuron_id = cin.neuron_id
            AND an.level = cin.level
            AND an.age = 0
        )
    `, [this.minErrorPatternThreshold]);

    if (failures.length === 0) return;

    // Group failures by level
    const failuresByLevel = new Map();
    for (const failure of failures) {
        if (!failuresByLevel.has(failure.level))
            failuresByLevel.set(failure.level, []);
        failuresByLevel.get(failure.level).push(failure);
    }

    // Create patterns at level+1 for each failed prediction
    for (const [level, levelFailures] of failuresByLevel) {
        await this.createErrorPatterns(levelFailures, level);
    }
}

async createErrorPatterns(failures, level) {
    // For each failed prediction, get the connections that caused it
    for (const failure of failures) {
        // Get connections from connection_inference that predicted this neuron
        const [connections] = await this.conn.query(`
            SELECT connection_id, strength
            FROM connection_inference
            WHERE level = ? AND to_neuron_id = ?
        `, [level, failure.neuron_id]);

        if (connections.length === 0) continue;

        // Create new pattern neuron at level+1
        const patternNeuronId = await this.bulkInsertNeurons(1);

        // Map pattern to peak
        await this.conn.query(
            'INSERT INTO pattern_peaks (pattern_neuron_id, peak_neuron_id) VALUES (?, ?)',
            [patternNeuronId[0], failure.neuron_id]
        );

        // Insert into pattern_past (connections leading TO the peak)
        // These are the connections that predicted the failed neuron
        const pastRows = connections.map(c => [patternNeuronId[0], c.connection_id, 1.0]);
        await this.conn.query(
            'INSERT INTO pattern_past (pattern_neuron_id, connection_id, strength) VALUES ?',
            [pastRows]
        );

        // Insert into pattern_future (connections FROM the peak)
        // Get connections where from_neuron_id = failure.neuron_id
        const [futureConnections] = await this.conn.query(`
            SELECT id as connection_id
            FROM connections
            WHERE from_neuron_id = ?
            AND distance > 0
            AND strength > 0
        `, [failure.neuron_id]);

        if (futureConnections.length > 0) {
            const futureRows = futureConnections.map(c => [patternNeuronId[0], c.connection_id, 1.0]);
            await this.conn.query(
                'INSERT INTO pattern_future (pattern_neuron_id, connection_id, strength) VALUES ?',
                [futureRows]
            );
        }
    }
}
```

### Pattern Future Unpacking Logic

**Pseudocode for unpacking:**
```javascript
async unpackPatternPredictions(predictions, targetLevel) {
    if (predictions.length === 0) return;

    // Get future connections from predicted neurons
    // predictions = [{ neuron_id, strength }, ...]
    const neuronIds = predictions.map(p => p.neuron_id);

    const [futureConnections] = await this.conn.query(`
        SELECT
            pf.pattern_neuron_id,
            c.to_neuron_id,
            c.distance,
            pf.strength * c.strength * POW(?, c.distance) as weighted_strength
        FROM pattern_future pf
        JOIN connections c ON pf.connection_id = c.id
        WHERE pf.pattern_neuron_id IN (?)
        AND c.strength > 0
    `, [this.peakTimeDecayFactor, neuronIds]);

    if (futureConnections.length === 0) return;

    // Aggregate by to_neuron_id
    const neuronStrengths = new Map();
    for (const conn of futureConnections) {
        if (!neuronStrengths.has(conn.to_neuron_id))
            neuronStrengths.set(conn.to_neuron_id, 0);
        neuronStrengths.set(conn.to_neuron_id,
            neuronStrengths.get(conn.to_neuron_id) + conn.weighted_strength);
    }

    // Calculate average
    const strengths = Array.from(neuronStrengths.values());
    const avgStrength = strengths.reduce((a, b) => a + b, 0) / strengths.length;

    // Filter above average
    const filteredNeurons = [];
    for (const [neuronId, strength] of neuronStrengths) {
        if (strength > avgStrength)
            filteredNeurons.push({ neuron_id: neuronId, strength, level: targetLevel });
    }

    if (filteredNeurons.length === 0) return;

    // Write to pattern_inferred_neurons
    const rows = filteredNeurons.map(n => [n.neuron_id, n.level, 0, n.strength]);
    await this.conn.query(
        'INSERT INTO pattern_inferred_neurons (neuron_id, level, age, strength) VALUES ?',
        [rows]
    );

    // Recurse down if not at base level
    if (targetLevel > 0)
        await this.unpackPatternPredictions(filteredNeurons, targetLevel - 1);
}
```

### Connection Inference Per Level

**Refactored `inferConnectionsAtLevel(level)`:**
```javascript
async inferConnectionsAtLevel(level) {
    // Get active neurons at this level
    const [activeNeurons] = await this.conn.query(
        'SELECT neuron_id, age FROM active_neurons WHERE level = ?',
        [level]
    );

    if (activeNeurons.length === 0) return;

    // Get candidate connections (same logic as current inferConnections but filtered by level)
    const [connections] = await this.conn.query(`
        SELECT
            c.id as connection_id,
            c.to_neuron_id,
            c.strength * POW(?, c.distance) as strength
        FROM active_neurons an
        JOIN connections c ON c.from_neuron_id = an.neuron_id
        WHERE an.level = ?
        AND c.distance = an.age + 1
        AND c.strength > 0
    `, [this.peakTimeDecayFactor, level]);

    if (connections.length === 0) return;

    // Aggregate and detect peaks (same logic as current implementation)
    // ... peak detection logic ...

    // Write to connection_inferred_neurons
    await this.conn.query(
        'INSERT INTO connection_inferred_neurons (neuron_id, level, age, strength) VALUES ?',
        [peakNeurons.map(p => [p.neuron_id, level, 0, p.strength])]
    );
}
```

### Pattern Inference From Level

**New method `inferPatternsFromLevel(level)`:**
```javascript
async inferPatternsFromLevel(level) {
    // Get active neurons at this level that are pattern neurons
    const [patternNeurons] = await this.conn.query(`
        SELECT an.neuron_id, an.age
        FROM active_neurons an
        JOIN pattern_peaks pp ON an.neuron_id = pp.pattern_neuron_id
        WHERE an.level = ?
        AND an.age = 0
    `, [level]);

    if (patternNeurons.length === 0) return [];

    // Get their future connections to predict level-1
    const neuronIds = patternNeurons.map(p => p.neuron_id);
    const [predictions] = await this.conn.query(`
        SELECT
            pf.pattern_neuron_id,
            c.to_neuron_id as neuron_id,
            pf.strength * c.strength * POW(?, c.distance) as strength
        FROM pattern_future pf
        JOIN connections c ON pf.connection_id = c.id
        WHERE pf.pattern_neuron_id IN (?)
        AND c.strength > 0
    `, [this.peakTimeDecayFactor, neuronIds]);

    // Aggregate by neuron_id
    const neuronStrengths = new Map();
    for (const pred of predictions) {
        if (!neuronStrengths.has(pred.neuron_id))
            neuronStrengths.set(pred.neuron_id, 0);
        neuronStrengths.set(pred.neuron_id,
            neuronStrengths.get(pred.neuron_id) + pred.strength);
    }

    return Array.from(neuronStrengths.entries()).map(([neuron_id, strength]) =>
        ({ neuron_id, strength }));
}
```

## Summary of Key Changes

### Files to Modify
1. **brain.js**
   - Add `minErrorPatternThreshold` hyperparameter
   - Update `processFrame()` to call `validateAndLearnFromErrors()`

2. **brain-mysql.js**
   - Modify `activateLevelPatterns()` - remove creation/reinforcement
   - Modify `activateNeurons()` - remove reinforcement call
   - Add `validateAndLearnFromErrors()` method
   - Add `createErrorPatterns()` method
   - Refactor `inferNeurons()` - top-down flow
   - Add `getMaxActiveLevel()` method
   - Refactor `inferConnections()` → `inferConnectionsAtLevel(level)`
   - Add `inferPatternsFromLevel(level)` method
   - Add `unpackPatternPredictions()` method
   - Update `matchObservedPatterns()` - use pattern_past
   - Update `runForgetCycle()` - handle both pattern tables

3. **brain-memory.js**
   - Same changes as brain-mysql.js for in-memory implementation

4. **db/db.sql**
   - Rename `patterns` → `pattern_past`
   - Create `pattern_future` table
   - Update indexes as needed

### Migration Script
Create `db/migrate-pattern-split.js`:
- Rename patterns table
- Create pattern_future table
- Populate pattern_future from pattern_past
- Verify data integrity

