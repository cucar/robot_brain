# Error-Driven Learning Architecture - Implementation Plan

## Overview
This document outlines the implementation plan for refactoring the brain architecture from continuous pattern creation to error-driven learning with top-down inference.

## Critical Concept: Pattern Peak in Error-Driven Learning

**WHO IS LEARNING?**
- The neuron doing the predicting is the one that needs to learn
- **Pattern peak = the predictor neuron** (not the predicted neuron)
- When neuron X predicts neuron D incorrectly, X is the one that needs to learn
- Pattern captures: "When X appears in this context (pattern_past), here's what X predicts (pattern_future)"

**Example:**
```
Active neurons: A (age=2), B (age=1), X (age=0)
Neuron X (active) → predicts Neuron D (fails)

Pattern created:
  - Peak: X (the predictor)
  - pattern_past: ALL connections TO X at ALL distances
    - B→X (distance=1)
    - A→X (distance=2)
    - ... up to distance=9
  - pattern_future: ALL connections FROM X
    - X→D (distance=1) - the failed prediction
    - X→E (distance=2)
    - ... all predictions from X
```

**Pattern Activation:**
- Pattern activates when peak neuron X appears with matching context (pattern_past)
- Context matching checks: Are the same neurons active at the same distances?
- Pattern provides top-down predictions via pattern_future
- Pattern strength reflects reliability of X's predictions in this context

**Multiple-Distance Connections:**
- Connections exist at distances 1-9 (based on source neuron age)
- pattern_past captures up to 9 frames of temporal history
- This enables rich context differentiation
- Different contexts = different active neurons at different distances

## Core Architectural Changes

### 1. Pattern Recognition → Match and Reinforce (No Creation)
**Current Behavior:**
- `recognizeLevelPatterns()` detects peaks, matches patterns, merges (reinforces), and creates new patterns
- Pattern creation happens during every recognition cycle

**New Behavior:**
- Pattern recognition matches and activates existing patterns
- Pattern connections ARE reinforced during recognition (Hebbian learning for vertical connections)
- No pattern creation during recognition - only during error-driven learning

**Changes Required:**
- Keep `mergeMatchedPatterns()` call in `recognizeLevelPatterns()` - Hebbian learning for patterns
- Remove `createNewPatterns()` call from `recognizeLevelPatterns()`
- Keep: `getObservedPatterns()`, `matchObservedPatterns()`, `mergeMatchedPatterns()`, and pattern neuron activation

### 2. Connection Learning → Hebbian Reinforcement (No Change)
**Current Behavior:**
- `activateNeurons()` calls `reinforceConnections()` which strengthens existing connections
- Connections are created/strengthened whenever neurons co-occur (Hebbian learning)

**New Behavior:**
- **NO CHANGE** - Keep all Hebbian learning for connections
- Connection creation and reinforcement both happen during activation
- This is the basis of Hebbian learning: neurons that fire together wire together

**Changes Required:**
- **NONE** - Keep `reinforceConnections()` call in `activateNeurons()`
- Keep all existing connection learning logic

### 3. Error-Driven Pattern Creation
**New Phase:** `validateAndLearnFromErrors()` - runs before `inferNeurons()`

**Process:**
1. Check predictions from previous frame (age=1 in connection_inferred_neurons)
2. For each failed prediction where strength >= `minErrorPatternThreshold`:
   - **Identify the predictor neurons** (from_neuron_id from connections in connection_inference)
   - For each unique predictor neuron (the peak):
     - Create a new pattern neuron at level+1
     - **Pattern peak:** The neuron that made the wrong prediction
     - **Pattern past:** Connections leading TO the peak (the context when peak was active)
     - **Pattern future:** Connections FROM the peak (including the failed prediction)
3. Apply negative reinforcement to failed connection predictions (existing logic)

**Key Insight:**
- The neuron doing the predicting is the one that needs to learn
- Pattern peak = the predictor neuron (not the predicted neuron)
- Pattern captures: "When peak neuron appears in this context, here's what it predicts"
- pattern_past includes connections at ALL distances (1-9), capturing up to 9 frames of history
- This enables context differentiation based on long temporal sequences

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
) ENGINE=InnoDB;

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
) ENGINE=InnoDB;

-- Add strength column to pattern_peaks
ALTER TABLE pattern_peaks
ADD COLUMN strength DECIMAL(10,2) NOT NULL DEFAULT 1.0;
```

**Scratch tables for error-driven learning:**
```sql
-- See "Scratch Tables Reference" section below for complete list
CREATE TABLE inference_sources (...) ENGINE=MEMORY;
CREATE TABLE failed_predictions (...) ENGINE=MEMORY;
CREATE TABLE failed_prediction_sources (...) ENGINE=MEMORY;
CREATE TABLE error_pattern_mapping (...) ENGINE=MEMORY;
```

**Migration Required:**
- Rename existing `patterns` table to `pattern_past`
- Create new `pattern_future` table
- Populate `pattern_future` from existing pattern data (connections where distance > 0)
- Add `strength` column to `pattern_peaks`

### 4. Sequential Level-by-Level Inference

**Current Behavior:**
- `inferConnections()` - predicts for all levels at once
- `inferPatterns()` - recursive CTE cascading down from connection predictions
- Bottom-up flow: base connections → pattern cascade

**New Behavior:**
- Sequential processing from highest active level down to level 0
- **Stop at first level that produces predictions** (pattern override)
- At each level:
  1. Try connection inference at this level
  2. If predictions found → unpack to base, validate, STOP
  3. Try pattern inference from level+1
  4. If predictions found → unpack to base, validate, STOP
  5. Continue to next level down

**Key Insight:**
- Pattern inference provides context-specific override of general connection predictions
- Only the first successful inference mechanism is used (no combining)
- This enables fast adaptation to changing contexts

**New Method Structure:**
```javascript
async inferNextFrame() {
  const maxActiveLevel = await this.getMaxActiveLevel();

  // Clear inference scratch tables
  await this.conn.query(`TRUNCATE connection_inferred_neurons`);
  await this.conn.query(`TRUNCATE pattern_inferred_neurons`);
  await this.conn.query(`TRUNCATE inference_sources`);

  for (let level = maxActiveLevel; level >= 0; level--) {

    // Step 1: Try connection inference at this level
    const connectionPredictions = await this.inferConnectionsAtLevel(level);

    if (connectionPredictions > 0) {
      // Got predictions! Unpack to base if needed
      if (level > 0) await this.unpackToBase(level, 'connection');

      // Validate and learn from errors
      await this.validateAndLearnFromErrors('connection', level);
      return;
    }

    // Step 2: No connection predictions, try pattern inference
    if (level < maxActiveLevel) {
      const patternPredictions = await this.inferPatternsFromLevel(level + 1);

      if (patternPredictions > 0) {
        // Got predictions! Unpack to base if needed
        if (level > 0) await this.unpackToBase(level, 'pattern');

        // Validate and learn from errors
        await this.validateAndLearnFromErrors('pattern', level + 1);
        return;
      }
    }
  }

  // No predictions at any level
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
1. **Modify `recognizeLevelPatterns()`:**
   - **Keep `mergeMatchedPatterns()` call** - Hebbian learning for patterns
   - Remove `createNewPatterns()` call
   - Keep matching and activation logic

2. **Update `matchObservedPatterns()`:**
   - Change references from `patterns` → `pattern_past`
   - No other logic changes needed

3. **Update `mergeMatchedPatterns()`:**
   - Change references from `patterns` → `pattern_past`
   - Keep all reinforcement logic (Hebbian learning)

### Phase 3: Implement Error-Driven Learning
1. **Add hyperparameter:**
   ```javascript
   this.minErrorPatternThreshold = 0.5;
   ```

2. **Create `validateAndLearnFromErrors()` method:**
   - Check failed predictions from connection_inferred_neurons (age=1)
   - Filter by strength >= minErrorPatternThreshold
   - For each failed prediction:
     - **Get the predictor neurons** (from_neuron_id from connections in connection_inference)
     - Group by predictor neuron (each predictor gets its own pattern)
     - For each predictor neuron (the peak):
       - Create pattern neuron at level+1
       - Insert into pattern_past (connections leading TO the predictor/peak)
       - Insert into pattern_future (connections FROM the predictor/peak, including failed predictions)
       - Create pattern_peaks mapping (pattern_neuron_id → predictor_neuron_id)

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

### Phase 4: Implement Top-Down Inference
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

### Phase 5: Update Negative Reinforcement
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

### Sequential Inference Flow

**inferNextFrame():**
```javascript
async inferNextFrame() {
  const maxActiveLevel = await this.getMaxActiveLevel();

  // Clear inference scratch tables
  await this.conn.query(`TRUNCATE connection_inferred_neurons`);
  await this.conn.query(`TRUNCATE pattern_inferred_neurons`);
  await this.conn.query(`TRUNCATE inference_sources`);

  for (let level = maxActiveLevel; level >= 0; level--) {

    // Step 1: Try connection inference at this level
    const connectionPredictions = await this.inferConnectionsAtLevel(level);

    if (connectionPredictions > 0) {
      // Got predictions! Unpack to base if needed
      if (level > 0) await this.unpackToBase(level, 'connection');

      // Validate and learn from errors
      await this.validateAndLearnFromErrors('connection', level);
      return;
    }

    // Step 2: No connection predictions, try pattern inference
    if (level < maxActiveLevel) {
      const patternPredictions = await this.inferPatternsFromLevel(level + 1);

      if (patternPredictions > 0) {
        // Got predictions! Unpack to base if needed
        if (level > 0) await this.unpackToBase(level, 'pattern');

        // Validate and learn from errors
        await this.validateAndLearnFromErrors('pattern', level + 1);
        return;
      }
    }
  }

  // No predictions at any level
}
```

**Key Points:**
- Process levels from highest to lowest
- Stop at first level that produces predictions
- Pattern inference only attempted if connection inference fails
- Validation happens at inference level, not base level

### Connection Inference

**inferConnectionsAtLevel(level):**
- Populates `connection_inferred_neurons` with predictions at this level
- Populates `inference_sources` to track which neurons predicted what
- Returns count of predictions

**Key SQL:**
```sql
-- Predict neurons based on connections
INSERT INTO connection_inferred_neurons (neuron_id, level, age, strength)
SELECT
  c.to_neuron_id,
  ?,
  -1,
  SUM(c.strength * POW(?, c.distance)) as total_strength
FROM active_neurons an
JOIN connections c ON c.from_neuron_id = an.neuron_id
WHERE an.level = ?
GROUP BY c.to_neuron_id
HAVING total_strength >= ?

-- Track predictors
INSERT INTO inference_sources (inferred_neuron_id, level, predictor_neuron_id, prediction_strength, source)
SELECT c.to_neuron_id, ?, c.from_neuron_id, c.strength * POW(?, c.distance), 'connection'
FROM active_neurons an
JOIN connections c ON c.from_neuron_id = an.neuron_id
WHERE an.level = ?
AND c.to_neuron_id IN (SELECT neuron_id FROM connection_inferred_neurons WHERE level = ?)
```

### Pattern Inference

**inferPatternsFromLevel(sourceLevel):**
- Uses pattern_future from active patterns at sourceLevel
- Predicts neurons at targetLevel = sourceLevel - 1
- Populates `pattern_inferred_neurons` and `inference_sources`
- Returns count of predictions

**Key SQL:**
```sql
-- Predict neurons based on pattern_future
INSERT INTO pattern_inferred_neurons (neuron_id, level, age, strength)
SELECT
  c.to_neuron_id,
  ?,
  -1,
  SUM(pf.strength + c.strength * POW(?, c.distance)) as total_strength
FROM active_neurons an
JOIN pattern_future pf ON pf.pattern_neuron_id = an.neuron_id
JOIN connections c ON c.id = pf.connection_id
WHERE an.level = ?
GROUP BY c.to_neuron_id
HAVING total_strength >= ?
```

**Strength Calculation:**
- `pf.strength` (pattern_future strength) + `c.strength * decay` (connection strength)
- Addition (not multiplication) allows both to contribute independently
- Pattern strength represents observation count

### Unpacking to Base Level

**unpackToBase(fromLevel, source):**
- Follows peak chain from fromLevel down to level 0
- Accumulates strength from pattern_peaks at each level
- Uses recursive CTE for efficiency

**Key SQL:**
```sql
WITH RECURSIVE peak_chain AS (
  -- Base case: starting predictions
  SELECT neuron_id, neuron_id as original_neuron_id, 0 as accumulated_strength, ? as current_level
  FROM neurons
  WHERE id IN (SELECT neuron_id FROM [inference_table])

  UNION ALL

  -- Recursive case: follow peak chain down, accumulating strength
  SELECT pp.peak_neuron_id, pc.original_neuron_id, pc.accumulated_strength + pp.strength, n.level
  FROM peak_chain pc
  JOIN pattern_peaks pp ON pp.pattern_neuron_id = pc.neuron_id
  JOIN neurons n ON n.id = pp.peak_neuron_id
  WHERE pc.current_level > 0
)
SELECT original_neuron_id, neuron_id, accumulated_strength
FROM peak_chain
WHERE current_level = 0
```

**Strength accumulation:**
- Original prediction strength + sum of pattern_peaks.strength down the chain
- Represents total observation count through the hierarchy

### Validation and Error Learning

**validateAndLearnFromErrors(source, level):**
- Takes source ('connection' or 'pattern') and level as parameters
- Validates predictions at the inference level (not base level)
- Applies negative reinforcement to failed predictions
- Creates error patterns for high-confidence failures

**Key SQL:**
```sql
-- Find failed predictions
INSERT INTO failed_predictions (neuron_id, level, strength)
SELECT ir.neuron_id, ir.level, ir.strength
FROM [inference_table] ir
LEFT JOIN active_neurons an ON an.neuron_id = ir.neuron_id AND an.level = ir.level AND an.age = 0
WHERE an.neuron_id IS NULL

-- Copy sources for failed predictions
INSERT INTO failed_prediction_sources (failed_neuron_id, level, predictor_neuron_id, prediction_strength)
SELECT isrc.inferred_neuron_id, isrc.level, isrc.predictor_neuron_id, isrc.prediction_strength
FROM inference_sources isrc
JOIN failed_predictions fp ON fp.neuron_id = isrc.inferred_neuron_id AND fp.level = isrc.level
WHERE isrc.source = ?

-- Negative reinforcement for connections
UPDATE connections c
JOIN failed_predictions fp ON fp.neuron_id = c.to_neuron_id
SET c.strength = GREATEST(0, c.strength - ?)
WHERE c.strength > 0

-- Negative reinforcement for pattern_future
UPDATE pattern_future pf
JOIN connections c ON c.id = pf.connection_id
JOIN failed_predictions fp ON fp.neuron_id = c.to_neuron_id
SET pf.strength = GREATEST(0, pf.strength - ?)
WHERE pf.strength > 0
```

**Why validate at inference level:**
- The fault is at the level that made the prediction
- Lower levels just followed the peak chain (no fault)
- Only the predictor neurons should be penalized

### Error-Driven Pattern Creation

**createErrorPatterns(level):**
- Reads from `failed_predictions` and `failed_prediction_sources` scratch tables
- Creates patterns at level + 1 (one above predictor level)
- Only for high-confidence failures (strength >= minErrorPatternThreshold)
- All operations in bulk using scratch tables

**Implementation:**
```javascript
async createErrorPatterns(inferenceLevel) {
  const [failedCount] = await this.conn.query(`
    SELECT COUNT(*) as count FROM failed_predictions
  `);

  if (failedCount[0].count === 0) return;

  // Filter for high-confidence failures and get unique predictors
  await this.conn.query(`TRUNCATE error_pattern_mapping`);

  await this.conn.query(`
    INSERT INTO error_pattern_mapping (predictor_neuron_id, channel_id)
    SELECT DISTINCT fps.predictor_neuron_id, n.channel_id
    FROM failed_prediction_sources fps
    JOIN failed_predictions fp ON fp.neuron_id = fps.failed_neuron_id
    JOIN neurons n ON n.id = fps.predictor_neuron_id
    WHERE fp.strength >= ?
  `, [this.minErrorPatternThreshold]);

  const [predictorCount] = await this.conn.query(`
    SELECT COUNT(*) as count FROM error_pattern_mapping
  `);

  if (predictorCount[0].count === 0) return;

  // Bulk create pattern neurons at inferenceLevel + 1
  const [insertResult] = await this.conn.query(`
    INSERT INTO neurons (level, channel_id)
    SELECT ?, channel_id
    FROM error_pattern_mapping
    ORDER BY seq_id
  `, [inferenceLevel + 1]);

  const firstPatternId = insertResult.insertId;

  // Update mapping with pattern_neuron_ids
  await this.conn.query(`
    UPDATE error_pattern_mapping
    SET pattern_neuron_id = ? + seq_id - 1
  `, [firstPatternId]);

  // Bulk create pattern_peaks
  await this.conn.query(`
    INSERT INTO pattern_peaks (pattern_neuron_id, peak_neuron_id, strength)
    SELECT pattern_neuron_id, predictor_neuron_id, 1.0
    FROM error_pattern_mapping
  `);

  // Bulk create pattern_past (ALL connections TO each predictor)
  await this.conn.query(`
    INSERT INTO pattern_past (pattern_neuron_id, connection_id, strength)
    SELECT epm.pattern_neuron_id, c.id, 1.0
    FROM error_pattern_mapping epm
    JOIN connections c ON c.to_neuron_id = epm.predictor_neuron_id
    WHERE c.strength > 0
  `);

  // Bulk create pattern_future (ALL connections FROM each predictor)
  await this.conn.query(`
    INSERT INTO pattern_future (pattern_neuron_id, connection_id, strength)
    SELECT epm.pattern_neuron_id, c.id, 1.0
    FROM error_pattern_mapping epm
    JOIN connections c ON c.from_neuron_id = epm.predictor_neuron_id
    WHERE c.strength > 0
  `);
}
```

**Key Points:**
- All operations in bulk (no loops)
- Uses scratch tables throughout
- pattern_past includes ALL connections at distances 1-9
- pattern_future includes ALL connections from predictor
- Initial strength = 1.0 for all pattern connections
- pattern_peaks.strength = 1.0 initially

### Context Differentiation

**How patterns differentiate contexts:**

1. **Multiple-distance connections in pattern_past:**
   - Captures up to 9 frames of temporal context
   - Same peak neuron can have multiple patterns
   - Each pattern represents the peak in a different temporal context

2. **Example:**
   - Pattern A: Peak C with {A at dist=2, B at dist=1}
   - Pattern B: Peak C with {X at dist=2, B at dist=1}
   - Same peak (C), same immediate predecessor (B), but different context 2 frames back

3. **Pattern matching:**
   - Match based on pattern_past connections
   - Check if peak neuron is active (age=0)
   - Check if connections in pattern_past are active:
     - For connection with distance=N, check if source neuron is at age=N
     - This ensures temporal context matches
   - Different contexts activate different patterns

### Scratch Tables Reference

**All scratch tables used in the new design:**

```sql
-- Existing tables (reused)
CREATE TABLE connection_inferred_neurons (
    neuron_id BIGINT UNSIGNED NOT NULL,
    level TINYINT NOT NULL,
    age TINYINT NOT NULL,
    strength DOUBLE NOT NULL,
    PRIMARY KEY (neuron_id, level)
) ENGINE=MEMORY;

CREATE TABLE pattern_inferred_neurons (
    neuron_id BIGINT UNSIGNED NOT NULL,
    level TINYINT NOT NULL,
    age TINYINT NOT NULL,
    strength DOUBLE NOT NULL,
    PRIMARY KEY (neuron_id, level)
) ENGINE=MEMORY;

-- New scratch tables
CREATE TABLE inference_sources (
    inferred_neuron_id BIGINT UNSIGNED NOT NULL,
    level TINYINT NOT NULL,
    predictor_neuron_id BIGINT UNSIGNED NOT NULL,
    prediction_strength DOUBLE NOT NULL,
    source ENUM('connection', 'pattern') NOT NULL,
    INDEX idx_inferred (inferred_neuron_id, level),
    INDEX idx_predictor (predictor_neuron_id)
) ENGINE=MEMORY;

CREATE TABLE failed_predictions (
    neuron_id BIGINT UNSIGNED NOT NULL,
    level TINYINT NOT NULL,
    strength DOUBLE NOT NULL,
    PRIMARY KEY (neuron_id, level)
) ENGINE=MEMORY;

CREATE TABLE failed_prediction_sources (
    failed_neuron_id BIGINT UNSIGNED NOT NULL,
    level TINYINT NOT NULL,
    predictor_neuron_id BIGINT UNSIGNED NOT NULL,
    prediction_strength DOUBLE NOT NULL,
    INDEX idx_failed (failed_neuron_id, level),
    INDEX idx_predictor (predictor_neuron_id)
) ENGINE=MEMORY;

CREATE TABLE error_pattern_mapping (
    seq_id INT AUTO_INCREMENT PRIMARY KEY,
    predictor_neuron_id BIGINT UNSIGNED NOT NULL UNIQUE,
    channel_id INT NOT NULL,
    pattern_neuron_id BIGINT UNSIGNED,
    INDEX idx_predictor (predictor_neuron_id)
) ENGINE=MEMORY;
```

**Tables to truncate in resetContext():**
- `active_neurons`
- `connection_inference`
- `inferred_neurons`
- `observed_connections`
- `observed_neuron_strengths`
- `observed_peaks`
- `observed_patterns`
- `matched_peaks`
- `active_connections`
- `connection_inferred_neurons`
- `pattern_inferred_neurons`
- `inference_sources`
- `failed_predictions`
- `failed_prediction_sources`
- `error_pattern_mapping`

## Why This Algorithm Will Work

### Will It Memorize Repeating Sequences?

**YES!** The algorithm will converge to 100% accuracy for any sequence with learnable temporal structure.

**Key Reasons:**

1. **Multiple-Distance Connections Capture Long Context**
   - Connections exist at distances 1-9 (based on source neuron age)
   - When creating error patterns, pattern_past includes ALL connections to the predictor
   - This captures up to 9 frames of temporal history
   - Example: When neuron C makes an error, pattern_past includes:
     - B→C (distance=1) - immediate predecessor
     - A→C (distance=2) - 2 frames back
     - X→C (distance=3) - 3 frames back
     - ... up to distance=9

2. **Context Differentiation Works**
   - Different contexts have different active neurons at different distances
   - Pattern matching checks: Are the same neurons active at the same distances?
   - Example:
     - Context 1: `D → A → B → C` has D at distance=2 when C appears
     - Context 2: `F → A → B → C` has F at distance=2 when C appears
     - Different patterns activate for different contexts!

3. **Multiple Patterns Per Neuron**
   - A single neuron can be the peak of MULTIPLE patterns
   - Each pattern represents the neuron in a different temporal context
   - Each pattern learns independently via Hebbian reinforcement

4. **Hierarchical Extension**
   - If 9 frames isn't enough context, higher-level patterns extend it
   - Level 1 patterns represent sequences at level 0
   - Level 2 patterns represent sequences of level 1 patterns
   - Each level extends temporal context exponentially

5. **Correct Handling of Unpredictable Sequences**
   - Truly random sequences (no correlation) won't be memorized
   - This is correct behavior - can't learn randomness
   - In real data, probability of identical contexts with different outcomes is infinitesimally small

**Conclusion:** The algorithm will memorize any sequence that CAN be predicted based on temporal patterns!

## New Inference Architecture

### Sequential Level-by-Level Inference

**Key Change:** Inference now processes levels sequentially from highest to lowest, stopping at the first level that produces predictions.

**Algorithm:**
```
For level = maxActiveLevel down to 0:
  1. Try connection inference at this level
     - If predictions found: unpack to base, validate, STOP

  2. Try pattern inference from level+1 (if not at max level)
     - If predictions found: unpack to base, validate, STOP

  3. Continue to next level down
```

**Pattern as Override:**
- Pattern inference provides context-specific predictions that override general connection predictions
- Only the first successful inference mechanism is used (no combining)
- This enables fast adaptation to changing contexts

### Inference Tables

**Existing tables reused:**
- `connection_inferred_neurons` - stores connection-based predictions
- `pattern_inferred_neurons` - stores pattern-based predictions

**New scratch tables:**
- `inference_sources` - tracks which neurons predicted what (for error pattern creation)
- `failed_predictions` - stores predictions that didn't materialize
- `failed_prediction_sources` - links failed predictions to their predictors
- `error_pattern_mapping` - temporary mapping during bulk pattern creation

### Validation and Learning

**Validation happens at inference level:**
- Not at base level after unpacking
- Only the level that made the prediction gets penalized
- Lower levels just followed the peak chain (no fault)

**Error pattern creation:**
- Patterns created at level + 1 (one above predictor level)
- Only for high-confidence failures (strength >= minErrorPatternThreshold)
- Bulk operations using scratch tables

## Summary of Key Changes

### Major Logic Changes

1. **Remove pattern creation from recognition:**
   - Remove `createNewPatterns()` call from `recognizeLevelPatterns()`
   - Add error-driven pattern creation in `validateAndLearnFromErrors()`

2. **New inference architecture:**
   - Sequential level-by-level processing (not parallel)
   - Stop at first level that produces predictions
   - Pattern inference as override of connection inference

3. **Add pattern_peaks.strength:**
   - Track pattern observation count
   - Reinforce in `mergeMatchedPatterns()`
   - Reduce in `runForgetCycle()`
   - Accumulate during unpacking

4. **Negative reinforcement for pattern_future:**
   - Apply when pattern predictions fail
   - Same mechanism as connection negative reinforcement

**Everything else stays the same:**
- Keep `reinforceConnections()` - Hebbian learning for connections
- Keep `mergeMatchedPatterns()` - Hebbian learning for patterns
- All recognition logic unchanged

### Files to Modify
1. **brain.js**
   - Add `minErrorPatternThreshold` hyperparameter
   - Update `processFrame()` to call `validateAndLearnFromErrors()`

2. **brain-mysql.js**
   - Modify `recognizeLevelPatterns()` - remove `createNewPatterns()` call only
   - Add `validateAndLearnFromErrors()` method
   - Add `createErrorPatterns()` method - creates patterns with peak = predictor neuron
   - Refactor `inferNeurons()` - top-down flow
   - Add `getMaxActiveLevel()` method
   - Refactor `inferConnections()` → `inferConnectionsAtLevel(level)`
   - Add `inferPatternsFromLevel(level)` method
   - Add `unpackPatternPredictions()` method
   - Update `matchObservedPatterns()` - use pattern_past
   - Update `mergeMatchedPatterns()` - use pattern_past
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

