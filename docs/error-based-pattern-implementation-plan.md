# Error-Based Pattern Implementation Plan

## Overview

Integrate error-based pattern creation with existing pattern infrastructure by:
1. Split `patterns` table into `patterns_past` and `patterns_future`
2. Modify pattern matching to use only past connections
3. Modify pattern inference to activate on past connections, predict future connections
4. Create patterns from prediction errors (false negatives)
5. Use peak detection on inferred neurons from future connections

---

## Phase 1: Database Schema Changes

### 1.1 Drop Old Pattern Table

```sql
DROP TABLE IF EXISTS patterns;
```

### 1.2 Create New Pattern Tables

```sql
-- Past connections define the context that activates a pattern
CREATE TABLE IF NOT EXISTS patterns_past (
    pattern_neuron_id BIGINT UNSIGNED NOT NULL,
    connection_id BIGINT UNSIGNED NOT NULL,
    strength DOUBLE NOT NULL DEFAULT 1.0,
    PRIMARY KEY (pattern_neuron_id, connection_id),
    FOREIGN KEY (pattern_neuron_id) REFERENCES neurons(id) ON DELETE CASCADE,
    FOREIGN KEY (connection_id) REFERENCES connections(id) ON DELETE CASCADE,
    INDEX idx_connection (connection_id),
    INDEX idx_pattern_strength (pattern_neuron_id, strength)
) ENGINE=MEMORY;

-- Future connections define what the pattern predicts
CREATE TABLE IF NOT EXISTS patterns_future (
    pattern_neuron_id BIGINT UNSIGNED NOT NULL,
    connection_id BIGINT UNSIGNED NOT NULL,
    strength DOUBLE NOT NULL DEFAULT 1.0,
    PRIMARY KEY (pattern_neuron_id, connection_id),
    FOREIGN KEY (pattern_neuron_id) REFERENCES neurons(id) ON DELETE CASCADE,
    FOREIGN KEY (connection_id) REFERENCES connections(id) ON DELETE CASCADE,
    INDEX idx_connection (connection_id),
    INDEX idx_pattern_strength (pattern_neuron_id, strength)
) ENGINE=MEMORY;
```

### 1.3 Keep Pattern Peaks Table (Unchanged)

```sql
-- Maps pattern neurons to their peak neurons (unchanged)
CREATE TABLE IF NOT EXISTS pattern_peaks (
    pattern_neuron_id BIGINT UNSIGNED NOT NULL,
    peak_neuron_id BIGINT UNSIGNED NOT NULL,
    PRIMARY KEY (pattern_neuron_id),
    FOREIGN KEY (pattern_neuron_id) REFERENCES neurons(id) ON DELETE CASCADE,
    FOREIGN KEY (peak_neuron_id) REFERENCES neurons(id) ON DELETE CASCADE,
    INDEX idx_peak (peak_neuron_id)
) ENGINE=MEMORY;
```

**Note:** Peak neuron represents the "decision point" - the most recent neuron before the prediction error occurred.

---

## Phase 2: Pattern Creation from Errors

### 2.1 Detect Prediction Errors

Add to `negativeReinforceConnections()` or create new method `createPatternsFromErrors()`:

```javascript
async createPatternsFromErrors() {
    if (this.debug) console.log('Creating patterns from prediction errors');
    
    // Find false negatives: neurons that activated but weren't predicted
    const [falseNegatives] = await this.conn.query(`
        SELECT an.neuron_id
        FROM active_neurons an
        WHERE an.level = 0 AND an.age = 0
          AND NOT EXISTS (
              SELECT 1 FROM connection_inferred_neurons cin
              WHERE cin.neuron_id = an.neuron_id AND cin.level = 0
          )
          AND NOT EXISTS (
              SELECT 1 FROM pattern_inferred_neurons pin
              WHERE pin.neuron_id = an.neuron_id AND pin.level = 0
          )
    `);
    
    if (falseNegatives.length === 0) {
        if (this.debug) console.log('No prediction errors - skipping pattern creation');
        return;
    }
    
    if (this.debug) console.log(`Found ${falseNegatives.length} prediction errors`);
    
    // Collect past connections (ALL active connections from ages 1 to baseNeuronMaxAge)
    await this.collectPastConnectionsFromErrors();
    
    // Collect future connections to ALL false negative neurons
    await this.collectFutureConnectionsFromErrors(falseNegatives);
    
    // Find peak neuron (most recent active neuron, age=1)
    const peakNeuronId = await this.findPeakNeuronForErrors();
    
    if (!peakNeuronId) {
        if (this.debug) console.log('No peak neuron found - skipping pattern creation');
        return;
    }
    
    // Populate observed_patterns with past connections (for matching)
    await this.populateObservedPatternsFromErrors(peakNeuronId);
    
    // Use existing infrastructure to match/merge/create patterns
    await this.matchObservedPatterns();
    await this.mergeMatchedPatterns();
    await this.createNewPatterns();
    
    // Add future connections to matched/new patterns
    await this.addFutureConnectionsToPatterns();
}
```

### 2.2 Collect Past Connections

```javascript
async collectPastConnectionsFromErrors() {
    // Truncate scratch table
    await this.conn.query('TRUNCATE pattern_creation_past');
    
    // Collect ALL active connections (ages 1 to baseNeuronMaxAge)
    // These represent the context that was active when the error occurred
    await this.conn.query(`
        INSERT INTO pattern_creation_past (connection_id)
        SELECT DISTINCT ac.connection_id
        FROM active_connections ac
        WHERE ac.level = 0
          AND ac.age > 0  -- Exclude age=0 (current frame)
    `);
}
```

### 2.3 Collect Future Connections

```javascript
async collectFutureConnectionsFromErrors(falseNegatives) {
    // Truncate scratch table
    await this.conn.query('TRUNCATE pattern_creation_future');
    
    // Collect connections to ALL false negative neurons
    // These represent what should have been predicted
    const neuronIds = falseNegatives.map(fn => fn.neuron_id);
    
    await this.conn.query(`
        INSERT INTO pattern_creation_future (connection_id)
        SELECT DISTINCT ac.connection_id
        FROM active_connections ac
        WHERE ac.level = 0
          AND ac.age = 0  -- Current frame
          AND ac.to_neuron_id IN (?)
    `, [neuronIds]);
}
```

### 2.4 Find Peak Neuron

```javascript
async findPeakNeuronForErrors() {
    // Peak = most recent active neuron (age=1)
    // This is the "decision point" before the error occurred
    const [result] = await this.conn.query(`
        SELECT neuron_id
        FROM active_neurons
        WHERE level = 0 AND age = 1
        LIMIT 1
    `);
    
    return result.length > 0 ? result[0].neuron_id : null;
}
```

### 2.5 Populate Observed Patterns

```javascript
async populateObservedPatternsFromErrors(peakNeuronId) {
    // Populate observed_patterns with past connections
    // This allows existing matchObservedPatterns() to work
    await this.conn.query('TRUNCATE observed_patterns');
    await this.conn.query('TRUNCATE observed_peaks');
    
    // Add peak
    await this.conn.query(`
        INSERT INTO observed_peaks (peak_neuron_id, total_strength, connection_count)
        SELECT ?, 1.0, COUNT(*)
        FROM pattern_creation_past
    `, [peakNeuronId]);
    
    // Add past connections
    await this.conn.query(`
        INSERT INTO observed_patterns (peak_neuron_id, connection_id)
        SELECT ?, connection_id
        FROM pattern_creation_past
    `, [peakNeuronId]);
}
```

### 2.6 Add Future Connections to Patterns

```javascript
async addFutureConnectionsToPatterns() {
    // Add future connections to all matched/created patterns
    // Uses matched_patterns table populated by matchObservedPatterns/createNewPatterns
    
    await this.conn.query(`
        INSERT INTO patterns_future (pattern_neuron_id, connection_id, strength)
        SELECT mp.pattern_neuron_id, pcf.connection_id, 1.0
        FROM matched_patterns mp
        CROSS JOIN pattern_creation_future pcf
        ON DUPLICATE KEY UPDATE strength = LEAST(?, strength + 1.0)
    `, [this.maxConnectionStrength]);
}
```

### 2.7 Create Scratch Tables

```sql
-- Scratch table for collecting past connections during error-based pattern creation
CREATE TABLE IF NOT EXISTS pattern_creation_past (
    connection_id BIGINT UNSIGNED NOT NULL,
    PRIMARY KEY (connection_id)
) ENGINE=MEMORY;

-- Scratch table for collecting future connections during error-based pattern creation
CREATE TABLE IF NOT EXISTS pattern_creation_future (
    connection_id BIGINT UNSIGNED NOT NULL,
    PRIMARY KEY (connection_id)
) ENGINE=MEMORY;
```

---

## Phase 3: Modify Pattern Matching

### 3.1 Update matchObservedPatterns()

Change to match only on **past connections**:

```javascript
async matchObservedPatterns() {
    if (this.debug) console.log('Matching observed patterns to known patterns');
    
    await this.conn.query('TRUNCATE matched_peaks');
    await this.conn.query('TRUNCATE matched_patterns');
    
    // Match based on PAST connections only
    // At least mergePatternThreshold (e.g., 50%) of pattern's past connections must be observed
    await this.conn.query(`
        INSERT INTO matched_patterns (peak_neuron_id, pattern_neuron_id)
        SELECT pp.peak_neuron_id, pp.pattern_neuron_id
        FROM pattern_peaks pp
        JOIN observed_peaks opk ON pp.peak_neuron_id = opk.peak_neuron_id
        JOIN patterns_past p ON pp.pattern_neuron_id = p.pattern_neuron_id
        LEFT JOIN observed_patterns op ON pp.peak_neuron_id = op.peak_neuron_id 
            AND op.connection_id = p.connection_id
        GROUP BY pp.peak_neuron_id, pp.pattern_neuron_id
        HAVING COUNT(DISTINCT CASE WHEN op.connection_id IS NOT NULL 
                THEN p.connection_id END) >= COUNT(DISTINCT p.connection_id) * ?
    `, [this.mergePatternThreshold]);
    
    await this.conn.query(`
        INSERT INTO matched_peaks (peak_neuron_id)
        SELECT DISTINCT peak_neuron_id
        FROM matched_patterns
    `);
    
    const [result] = await this.conn.query('SELECT COUNT(*) as match_count FROM matched_patterns');
    if (this.debug) console.log(`Matched ${result[0].match_count} pattern-peak pairs`);
}
```

---

## Phase 4: Modify Pattern Merging

### 4.1 Update mergeMatchedPatterns()

Merge both past and future connections separately:

```javascript
async mergeMatchedPatterns() {
    if (this.debug) console.log('Merging matched patterns');
    
    // === PAST CONNECTIONS ===
    
    // Positive reinforcement: Add/strengthen observed past connections
    await this.conn.query(`
        INSERT INTO patterns_past (pattern_neuron_id, connection_id, strength)
        SELECT DISTINCT mp.pattern_neuron_id, op.connection_id, 1.0
        FROM matched_patterns mp
        JOIN observed_patterns op ON mp.peak_neuron_id = op.peak_neuron_id
        ON DUPLICATE KEY UPDATE strength = LEAST(?, strength + 1.0)
    `, [this.maxConnectionStrength]);
    
    // Negative reinforcement: Weaken unobserved past connections
    await this.conn.query(`
        UPDATE patterns_past p
        JOIN matched_patterns mp ON p.pattern_neuron_id = mp.pattern_neuron_id
        SET p.strength = GREATEST(?, LEAST(?, p.strength - ?))
        WHERE NOT EXISTS (
            SELECT 1 FROM observed_patterns op
            WHERE op.peak_neuron_id = mp.peak_neuron_id
              AND op.connection_id = p.connection_id
        )
    `, [this.minConnectionStrength, this.maxConnectionStrength, this.patternNegativeReinforcement]);
    
    // === FUTURE CONNECTIONS ===
    // (Only if pattern_creation_future has data - i.e., from error-based creation)
    
    const [futureCount] = await this.conn.query('SELECT COUNT(*) as count FROM pattern_creation_future');
    if (futureCount[0].count === 0) return;
    
    // Add/strengthen future connections from pattern_creation_future
    await this.conn.query(`
        INSERT INTO patterns_future (pattern_neuron_id, connection_id, strength)
        SELECT mp.pattern_neuron_id, pcf.connection_id, 1.0
        FROM matched_patterns mp
        CROSS JOIN pattern_creation_future pcf
        ON DUPLICATE KEY UPDATE strength = LEAST(?, strength + 1.0)
    `, [this.maxConnectionStrength]);
}
```

---

## Phase 5: Modify Pattern Creation

### 5.1 Update createNewPatterns()

Create patterns with both past and future connections:

```javascript
async createNewPatterns() {
    if (this.debug) console.log('Creating new patterns');
    
    const [peaksNeedingPatterns] = await this.conn.query(`
        SELECT opk.peak_neuron_id
        FROM observed_peaks opk
        WHERE NOT EXISTS (
            SELECT 1 FROM matched_peaks mpk
            WHERE mpk.peak_neuron_id = opk.peak_neuron_id
        )
    `);
    
    const count = peaksNeedingPatterns.length;
    if (this.debug) console.log(`Creating ${count} new patterns`);
    if (count === 0) return;
    
    const patternNeuronIds = await this.bulkInsertNeurons(count);
    
    const patternPeakMappings = [];
    for (let i = 0; i < count; i++)
        patternPeakMappings.push([patternNeuronIds[i], peaksNeedingPatterns[i].peak_neuron_id]);
    
    await this.conn.query(
        'INSERT INTO pattern_peaks (pattern_neuron_id, peak_neuron_id) VALUES ?',
        [patternPeakMappings]
    );
    
    // Insert PAST connections
    await this.conn.query(`
        INSERT INTO patterns_past (pattern_neuron_id, connection_id, strength)
        SELECT pp.pattern_neuron_id, op.connection_id, 1.0
        FROM pattern_peaks pp
        JOIN observed_patterns op ON pp.peak_neuron_id = op.peak_neuron_id
        WHERE pp.pattern_neuron_id IN (?)
    `, [patternNeuronIds]);
    
    // Insert FUTURE connections (if any - from error-based creation)
    const [futureCount] = await this.conn.query('SELECT COUNT(*) as count FROM pattern_creation_future');
    if (futureCount[0].count > 0) {
        await this.conn.query(`
            INSERT INTO patterns_future (pattern_neuron_id, connection_id, strength)
            SELECT pp.pattern_neuron_id, pcf.connection_id, 1.0
            FROM pattern_peaks pp
            CROSS JOIN pattern_creation_future pcf
            WHERE pp.pattern_neuron_id IN (?)
        `, [patternNeuronIds]);
    }
    
    // Add to matched_patterns for activation
    await this.conn.query(`
        INSERT INTO matched_patterns (pattern_neuron_id, peak_neuron_id)
        SELECT pattern_neuron_id, peak_neuron_id
        FROM pattern_peaks
        WHERE pattern_neuron_id IN (?)
    `, [patternNeuronIds]);
}
```

---

## Phase 6: Modify Pattern Inference

### 6.1 Complete Redesign of inferPatterns()

**Old approach:** Hierarchical cascade (pattern neuron → peak neuron at level below)

**New approach:** 
1. Check which patterns should activate (based on past connections)
2. Predict future connections from activated patterns
3. Aggregate predictions by to_neuron_id
4. Apply peak detection to filter predictions

```javascript
async inferPatterns() {
    if (this.debug) console.log('Inferring patterns');
    
    // Step 1: Find patterns whose past connections are ALL active
    await this.findActivatedPatterns();
    
    // Step 2: Predict future connections from activated patterns
    await this.predictFromActivatedPatterns();
    
    // Step 3: Aggregate predictions by neuron and apply peak detection
    await this.filterPatternPredictions();
}
```

### 6.2 Find Activated Patterns

```javascript
async findActivatedPatterns() {
    // Truncate scratch table
    await this.conn.query('TRUNCATE activated_patterns');
    
    // Find patterns where ALL past connections are currently active
    // Uses pattern activation threshold (default 1.0 = 100%)
    await this.conn.query(`
        INSERT INTO activated_patterns (pattern_neuron_id, activation_strength)
        SELECT 
            pp.pattern_neuron_id,
            COUNT(DISTINCT CASE WHEN ac.connection_id IS NOT NULL THEN pp.connection_id END) / 
                COUNT(DISTINCT pp.connection_id) as activation_ratio
        FROM patterns_past pp
        LEFT JOIN active_connections ac ON pp.connection_id = ac.connection_id AND ac.level = 0
        GROUP BY pp.pattern_neuron_id
        HAVING activation_ratio >= ?
    `, [this.patternActivationThreshold]);
    
    const [result] = await this.conn.query('SELECT COUNT(*) as count FROM activated_patterns');
    if (this.debug) console.log(`Activated ${result[0].count} patterns`);
}
```

### 6.3 Predict Future Connections

```javascript
async predictFromActivatedPatterns() {
    // For each activated pattern, predict its future connections
    // Aggregate by to_neuron_id (multiple connections can point to same neuron)
    
    await this.conn.query('TRUNCATE pattern_predicted_neurons');
    
    await this.conn.query(`
        INSERT INTO pattern_predicted_neurons (neuron_id, total_strength)
        SELECT 
            c.to_neuron_id as neuron_id,
            SUM(pf.strength * ap.activation_strength) as total_strength
        FROM activated_patterns ap
        JOIN patterns_future pf ON ap.pattern_neuron_id = pf.pattern_neuron_id
        JOIN connections c ON pf.connection_id = c.id
        GROUP BY c.to_neuron_id
    `);
    
    const [result] = await this.conn.query('SELECT COUNT(*) as count FROM pattern_predicted_neurons');
    if (this.debug) console.log(`Patterns predicted ${result[0].count} neurons (before filtering)`);
}
```

### 6.4 Filter Pattern Predictions

```javascript
async filterPatternPredictions() {
    // Apply peak detection: keep only neurons above average strength
    // If only one neuron predicted, keep it regardless of strength
    
    const [stats] = await this.conn.query(`
        SELECT COUNT(*) as count, AVG(total_strength) as avg_strength
        FROM pattern_predicted_neurons
    `);
    
    const count = stats[0].count;
    const avgStrength = stats[0].avg_strength || 0;
    
    if (count === 0) {
        if (this.debug) console.log('No pattern predictions');
        return;
    }
    
    // If only one prediction, keep it
    if (count === 1) {
        await this.conn.query(`
            INSERT INTO pattern_inferred_neurons (neuron_id, level, age, strength)
            SELECT neuron_id, 0, 0, total_strength
            FROM pattern_predicted_neurons
        `);
        if (this.debug) console.log('Kept 1 pattern prediction (only one)');
        return;
    }
    
    // Multiple predictions: keep only above-average
    await this.conn.query(`
        INSERT INTO pattern_inferred_neurons (neuron_id, level, age, strength)
        SELECT neuron_id, 0, 0, total_strength
        FROM pattern_predicted_neurons
        WHERE total_strength > ?
    `, [avgStrength]);
    
    const [result] = await this.conn.query('SELECT COUNT(*) as count FROM pattern_inferred_neurons');
    if (this.debug) console.log(`Kept ${result[0].count} pattern predictions (above average)`);
}
```

### 6.5 Create Scratch Tables

```sql
-- Patterns that are activated in current frame
CREATE TABLE IF NOT EXISTS activated_patterns (
    pattern_neuron_id BIGINT UNSIGNED NOT NULL,
    activation_strength DOUBLE NOT NULL DEFAULT 1.0,
    PRIMARY KEY (pattern_neuron_id)
) ENGINE=MEMORY;

-- Neurons predicted by activated patterns (before filtering)
CREATE TABLE IF NOT EXISTS pattern_predicted_neurons (
    neuron_id BIGINT UNSIGNED NOT NULL,
    total_strength DOUBLE NOT NULL,
    PRIMARY KEY (neuron_id)
) ENGINE=MEMORY;
```

---

## Phase 7: Update Forget Cycle

### 7.1 Modify runForgetCycle()

Update to handle both past and future pattern tables:

```javascript
// Delete weak past connections
await this.conn.query('DELETE FROM patterns_past WHERE strength < ?', [this.minConnectionStrength]);

// Delete weak future connections
await this.conn.query('DELETE FROM patterns_future WHERE strength < ?', [this.minConnectionStrength]);

// Delete pattern neurons with no connections
await this.conn.query(`
    DELETE n FROM neurons n
    WHERE NOT EXISTS (SELECT 1 FROM patterns_past pp WHERE pp.pattern_neuron_id = n.id)
      AND NOT EXISTS (SELECT 1 FROM patterns_future pf WHERE pf.pattern_neuron_id = n.id)
      AND EXISTS (SELECT 1 FROM pattern_peaks ppk WHERE ppk.pattern_neuron_id = n.id)
`);
```

---

## Phase 8: Add Hyperparameters

```javascript
// In brain.js constructor
this.patternActivationThreshold = 1.0;  // 100% of past connections must be active
```

---

## Phase 9: Integration Points

### 9.1 Call Pattern Creation from processFrame()

Add after `negativeReinforceConnections()`:

```javascript
// Create patterns from prediction errors
await this.createPatternsFromErrors();
```

### 9.2 Ensure Proper Order

```javascript
async processFrame() {
    // ... existing code ...
    
    // Inference
    await this.inferConnections();
    await this.inferPatterns();  // NEW: Uses past/future connections
    await this.resolveConflicts();
    
    // Recognition & Learning
    await this.activatePatternNeurons();  // Existing: hierarchical pattern recognition
    await this.reinforceConnections();
    await this.negativeReinforceConnections();
    await this.createPatternsFromErrors();  // NEW: Error-based pattern creation
    
    // ... rest of frame processing ...
}
```

---

## Summary of Changes

### Database
- ✅ Split `patterns` → `patterns_past` + `patterns_future`
- ✅ Add scratch tables: `pattern_creation_past`, `pattern_creation_future`, `activated_patterns`, `pattern_predicted_neurons`

### Methods Modified
- ✅ `matchObservedPatterns()` - Match on past connections only
- ✅ `mergeMatchedPatterns()` - Merge both past and future separately
- ✅ `createNewPatterns()` - Create with both past and future
- ✅ `inferPatterns()` - Complete redesign (activate on past, predict future)
- ✅ `runForgetCycle()` - Handle both tables

### Methods Added
- ✅ `createPatternsFromErrors()` - Main entry point
- ✅ `collectPastConnectionsFromErrors()` - Collect context
- ✅ `collectFutureConnectionsFromErrors()` - Collect predictions
- ✅ `findPeakNeuronForErrors()` - Find decision point
- ✅ `populateObservedPatternsFromErrors()` - Prepare for matching
- ✅ `addFutureConnectionsToPatterns()` - Add predictions to patterns
- ✅ `findActivatedPatterns()` - Check activation conditions
- ✅ `predictFromActivatedPatterns()` - Generate predictions
- ✅ `filterPatternPredictions()` - Apply peak detection

### Hyperparameters Added
- ✅ `patternActivationThreshold` - Fraction of past connections needed (default 1.0)

**Ready for implementation!** 🚀

