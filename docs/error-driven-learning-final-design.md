# Error-Driven Learning - Final Design

## Overview

This document describes the final design for error-driven pattern learning with sequential level-by-level inference.

## Key Principles

1. **Pattern as Override**: Pattern inference overrides connection inference for fast adaptation
2. **Sequential Processing**: Process levels from highest to lowest, stop at first success
3. **Validation at Inference Level**: Only the level that made the prediction gets penalized
4. **Scratch Tables**: All data flow through pre-created MEMORY tables
5. **Bulk Operations**: No loops, all SQL-based operations

## Inference Algorithm

### Sequential Level-by-Level Processing

```
For level = maxActiveLevel down to 0:
  1. Try connection inference at this level
     - If predictions found: unpack to base (if needed), validate, STOP
  
  2. Try pattern inference from level+1 (if not at max level)
     - If predictions found: unpack to base (if needed), validate, STOP
  
  3. Continue to next level down
```

### Why Stop at First Success?

- **Pattern override**: Patterns provide context-specific predictions that override general connections
- **Fast adaptation**: Enables instant learning when confident predictions fail
- **No combining**: Only one inference mechanism used per frame (cleaner semantics)

## Method Signatures

All methods use scratch tables for data flow (no parameters for data):

### inferNextFrame()
- **Input**: None (reads from active_neurons)
- **Output**: None (writes to connection_inferred_neurons or pattern_inferred_neurons)
- **Side effects**: Calls validateAndLearnFromErrors with source and level

### inferConnectionsAtLevel(level)
- **Input**: level (which level to infer at)
- **Output**: count of predictions
- **Writes to**: connection_inferred_neurons, inference_sources

### inferPatternsFromLevel(sourceLevel)
- **Input**: sourceLevel (level where patterns are, predicts targetLevel = sourceLevel - 1)
- **Output**: count of predictions
- **Writes to**: pattern_inferred_neurons, inference_sources

### saveInferenceChain(fromLevel, source)
- **Input**: fromLevel (level where predictions are), source ('connection' or 'pattern')
- **Output**: None
- **Reads from**: connection_inferred_neurons or pattern_inferred_neurons
- **Writes to**: Same table, updates with base level predictions

### validateAndLearnFromErrors(source, level)
- **Input**: source ('connection' or 'pattern'), level (where inference was made)
- **Output**: None
- **Reads from**: connection_inferred_neurons or pattern_inferred_neurons, inference_sources
- **Writes to**: failed_predictions, failed_prediction_sources
- **Side effects**: Applies negative reinforcement, calls createErrorPatterns

### createErrorPatterns(inferenceLevel)
- **Input**: inferenceLevel (where predictors are)
- **Output**: None
- **Reads from**: failed_predictions, failed_prediction_sources
- **Writes to**: neurons, pattern_peaks, pattern_past, pattern_future
- **Creates patterns at**: inferenceLevel + 1

## Scratch Tables

### Existing Tables (Reused)

```sql
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
```

### New Scratch Tables

```sql
-- Original inference sources (at the level where inference was made)
CREATE TABLE org_inference_sources (
    age TINYINT UNSIGNED NOT NULL DEFAULT 0,
    inferred_neuron_id BIGINT UNSIGNED NOT NULL,
    level TINYINT NOT NULL,
    source_type ENUM('connection', 'pattern') NOT NULL,
    source_id BIGINT UNSIGNED NOT NULL,
    inference_strength DOUBLE NOT NULL,
    PRIMARY KEY (age, inferred_neuron_id, level, source_type, source_id)
);

-- Base inference sources (unpacked to base level for rewards)
CREATE TABLE base_inference_sources (
    age TINYINT UNSIGNED NOT NULL DEFAULT 0,
    base_neuron_id BIGINT UNSIGNED NOT NULL,
    source_type ENUM('connection', 'pattern') NOT NULL,
    source_id BIGINT UNSIGNED NOT NULL,
    inference_strength DOUBLE NOT NULL,
    PRIMARY KEY (age, base_neuron_id, source_type, source_id),
    INDEX idx_base_age (base_neuron_id, age)
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

## Schema Changes

### New Persistent Tables

```sql
CREATE TABLE pattern_past (
    pattern_neuron_id BIGINT UNSIGNED NOT NULL,
    connection_id BIGINT UNSIGNED NOT NULL,
    strength DECIMAL(10,2) NOT NULL DEFAULT 1.0,
    PRIMARY KEY (pattern_neuron_id, connection_id),
    FOREIGN KEY (pattern_neuron_id) REFERENCES neurons(id) ON DELETE CASCADE,
    FOREIGN KEY (connection_id) REFERENCES connections(id) ON DELETE CASCADE
) ENGINE=InnoDB;

CREATE TABLE pattern_future (
    pattern_neuron_id BIGINT UNSIGNED NOT NULL,
    connection_id BIGINT UNSIGNED NOT NULL,
    strength DECIMAL(10,2) NOT NULL DEFAULT 1.0,
    PRIMARY KEY (pattern_neuron_id, connection_id),
    FOREIGN KEY (pattern_neuron_id) REFERENCES neurons(id) ON DELETE CASCADE,
    FOREIGN KEY (connection_id) REFERENCES connections(id) ON DELETE CASCADE
) ENGINE=InnoDB;
```

### Modify Existing Table

```sql
ALTER TABLE pattern_peaks
ADD COLUMN strength DECIMAL(10,2) NOT NULL DEFAULT 1.0;
```

## Data Flow

### Inference Flow

```
active_neurons (level N)
  ↓
inferConnectionsAtLevel(N)
  ↓
connection_inferred_neurons (level N)
inference_sources (source='connection')
  ↓
saveInferenceChain(N, 'connection')  [if N > 0]
  ↓
connection_inferred_neurons (level 0)
  ↓
validateAndLearnFromErrors('connection', N)
  ↓
failed_predictions
failed_prediction_sources
  ↓
createErrorPatterns(N)
  ↓
neurons (level N+1)
pattern_peaks
pattern_past
pattern_future
```

### Pattern Inference Flow

```
active_neurons (level N)
  ↓
inferPatternsFromLevel(N)
  ↓
pattern_inferred_neurons (level N-1)
inference_sources (source='pattern')
  ↓
saveInferenceChain(N-1, 'pattern')  [if N-1 > 0]
  ↓
pattern_inferred_neurons (level 0)
  ↓
validateAndLearnFromErrors('pattern', N)
  ↓
[same as above]
```

## Validation and Learning

### Validation at Inference Level

**Why validate at inference level, not base level?**
- The fault is at the level that made the prediction
- Lower levels just followed the peak chain (no fault)
- Only the predictor neurons should be penalized

### Negative Reinforcement

**For connection inference:**
```sql
UPDATE connections c
JOIN failed_predictions fp ON fp.neuron_id = c.to_neuron_id
SET c.strength = GREATEST(0, c.strength - ?)
WHERE c.strength > 0
```

**For pattern inference:**
```sql
UPDATE pattern_future pf
JOIN connections c ON c.id = pf.connection_id
JOIN failed_predictions fp ON fp.neuron_id = c.to_neuron_id
SET pf.strength = GREATEST(0, pf.strength - ?)
WHERE pf.strength > 0
```

## Error Pattern Creation

### High-Confidence Failures Only

Only create patterns when prediction strength >= minErrorPatternThreshold

### Bulk Operations

All operations done in bulk using scratch tables:
1. Filter high-confidence failures → error_pattern_mapping
2. Bulk create pattern neurons at inferenceLevel + 1
3. Bulk create pattern_peaks (strength = 1.0)
4. Bulk create pattern_past (ALL connections TO predictor)
5. Bulk create pattern_future (ALL connections FROM predictor)

### Pattern Structure

**Pattern peak**: The predictor neuron (not the predicted neuron)

**Pattern past**: ALL connections TO the predictor at distances 1-9
- Captures up to 9 frames of temporal context
- Enables context differentiation

**Pattern future**: ALL connections FROM the predictor
- Includes the failed prediction
- Includes all other predictions from this neuron

## Context Truncation in resetContext()

```javascript
async resetContext() {
  await this.truncateTables([
    'active_neurons',
    'connection_inference',
    'inferred_neurons',
    'observed_connections',
    'observed_neuron_strengths',
    'observed_peaks',
    'observed_patterns',
    'matched_peaks',
    'active_connections',
    'connection_inferred_neurons',
    'pattern_inferred_neurons',
    'inference_sources',
    'failed_predictions',
    'failed_prediction_sources',
    'error_pattern_mapping'
  ]);
}
```

## Summary of Changes

### Major Logic Changes

1. **Remove pattern creation from recognition**
   - Remove createNewPatterns() from recognizeLevelPatterns()
   - Keep mergeMatchedPatterns() for Hebbian learning

2. **New sequential inference architecture**
   - Process levels from highest to lowest
   - Stop at first level with predictions
   - Pattern inference as override of connection inference

3. **Add pattern_peaks.strength**
   - Track pattern observation count
   - Reinforce in mergeMatchedPatterns()
   - Reduce in runForgetCycle()
   - Accumulate during unpacking

4. **Negative reinforcement for pattern_future**
   - Apply when pattern predictions fail
   - Same mechanism as connection negative reinforcement

5. **Error-driven pattern creation**
   - Only for high-confidence failures
   - Bulk operations using scratch tables
   - Patterns created at level + 1

### No Changes

- Keep reinforceConnections() - Hebbian learning for connections
- Keep mergeMatchedPatterns() - Hebbian learning for patterns
- All recognition logic unchanged

