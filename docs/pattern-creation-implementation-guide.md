# Error-Based Pattern Creation: Implementation Guide

## Database Schema Changes

### 1. Modify `patterns` table

```sql
-- Add connection_type column to distinguish past vs future connections
ALTER TABLE patterns
ADD COLUMN connection_type ENUM('past', 'future') NOT NULL AFTER connection_id;

-- Update primary key to include connection_type
ALTER TABLE patterns
DROP PRIMARY KEY,
ADD PRIMARY KEY (pattern_neuron_id, connection_id, connection_type);

-- Add index for efficient filtering by type
ALTER TABLE patterns
ADD INDEX idx_pattern_type (pattern_neuron_id, connection_type);
```

### 2. Add scratch tables for pattern creation

```sql
-- Temporary table for past connections during pattern creation
CREATE TABLE IF NOT EXISTS pattern_creation_past (
    connection_id BIGINT UNSIGNED NOT NULL,
    strength DOUBLE NOT NULL,
    PRIMARY KEY (connection_id)
) ENGINE=MEMORY;

-- Temporary table for future connections during pattern creation
CREATE TABLE IF NOT EXISTS pattern_creation_future (
    connection_id BIGINT UNSIGNED NOT NULL,
    from_neuron_id BIGINT UNSIGNED NOT NULL,
    to_neuron_id BIGINT UNSIGNED NOT NULL,
    distance TINYINT UNSIGNED NOT NULL,
    PRIMARY KEY (connection_id)
) ENGINE=MEMORY;
```

---

## Implementation Steps

### Step 1: Modify `negativeReinforceConnections()`

Add pattern creation for false negatives:

```javascript
async negativeReinforceConnections() {
    
    // === EXISTING CODE: Handle false positives ===
    
    // Find which predictions failed (not in active_connections) across all levels
    const [failures] = await this.conn.query(`
        SELECT ci.level, ci.connection_id
        FROM connection_inference ci
        WHERE NOT EXISTS (
            SELECT 1 
            FROM active_connections ac
            WHERE ci.connection_id = ac.connection_id 
              AND ac.level = ci.level 
              AND ac.age = 0
        )    
    `);

    if (failures.length > 0) {
        // Apply negative reinforcement to failed predictions
        const failedConnectionIds = failures.map(f => f.connection_id);
        await this.conn.query(
            'UPDATE connections SET strength = GREATEST(?, LEAST(?, strength - ?)) WHERE id IN (?)',
            [this.minConnectionStrength, this.maxConnectionStrength, this.negativeLearningRate, failedConnectionIds]);
    }
    
    // === NEW CODE: Handle false negatives ===
    
    // Find neurons that activated but weren't predicted (false negatives)
    // Only at level 0 for now
    const [falseNegatives] = await this.conn.query(`
        SELECT an.neuron_id
        FROM active_neurons an
        WHERE an.level = 0
          AND an.age = 0
          AND NOT EXISTS (
              SELECT 1 FROM connection_inferred_neurons cin
              WHERE cin.neuron_id = an.neuron_id AND cin.level = 0 AND cin.age = 0
          )
          AND NOT EXISTS (
              SELECT 1 FROM pattern_inferred_neurons pin
              WHERE pin.neuron_id = an.neuron_id AND pin.level = 0 AND pin.age = 0
          )
    `);
    
    // Create patterns for each false negative
    for (const fn of falseNegatives) {
        await this.createPatternFromError(fn.neuron_id);
    }
    
    if (this.debug && falseNegatives.length > 0) {
        console.log(`Created ${falseNegatives.length} patterns from false negative predictions`);
    }
}
```

---

### Step 2: Implement `createPatternFromError()`

```javascript
/**
 * Create pattern from prediction error (false negative).
 * Pattern captures the context (past connections) that led to the error
 * and the correction (future connections) to predict the missed neuron.
 */
async createPatternFromError(neuronId) {
    
    // Step 1: Determine peak neuron (most recent active neuron at age=1)
    const [peakResult] = await this.conn.query(`
        SELECT neuron_id
        FROM active_neurons
        WHERE level = 0 AND age = 1
        ORDER BY neuron_id
        LIMIT 1
    `);
    
    if (peakResult.length === 0) {
        // No peak neuron available (not enough context)
        if (this.debug) console.log(`Cannot create pattern for N${neuronId}: no peak neuron (age=1)`);
        return;
    }
    
    const peakNeuronId = peakResult[0].neuron_id;
    
    // Step 2: Collect past connections (context)
    await this.collectPastConnections();
    
    // Step 3: Collect/create future connections (prediction)
    await this.collectFutureConnections(neuronId);
    
    // Step 4: Check for existing patterns to merge with
    const existingPatternId = await this.findMatchingPattern(peakNeuronId);
    
    if (existingPatternId) {
        // Merge: add future connections to existing pattern
        await this.mergePatternConnections(existingPatternId);
        if (this.debug) console.log(`Merged pattern for N${neuronId} into existing pattern P${existingPatternId}`);
    } else {
        // Create new pattern
        const newPatternId = await this.createNewPatternFromScratch(peakNeuronId);
        if (this.debug) console.log(`Created new pattern P${newPatternId} for N${neuronId} (peak: N${peakNeuronId})`);
    }
    
    // Step 5: Clean up scratch tables
    await this.conn.query('TRUNCATE pattern_creation_past');
    await this.conn.query('TRUNCATE pattern_creation_future');
}
```

---

### Step 3: Implement `collectPastConnections()`

```javascript
/**
 * Collect all active connections in the temporal window (ages 1 to patternCreationWindow).
 * These represent the "context" that should trigger the pattern.
 */
async collectPastConnections() {
    await this.conn.query(`
        INSERT INTO pattern_creation_past (connection_id, strength)
        SELECT ac.connection_id, c.strength
        FROM active_connections ac
        JOIN connections c ON ac.connection_id = c.id
        WHERE ac.level = 0
          AND ac.age BETWEEN 1 AND ?
    `, [this.patternCreationWindow]);
}
```

---

### Step 4: Implement `collectFutureConnections()`

```javascript
/**
 * Collect/create all connections to the unpredicted neuron from active neurons.
 * These represent the "prediction" the pattern should make.
 */
async collectFutureConnections(toNeuronId) {
    
    // Get all active neurons in the window
    const [activeNeurons] = await this.conn.query(`
        SELECT neuron_id, age
        FROM active_neurons
        WHERE level = 0
          AND age BETWEEN 0 AND ?
    `, [this.patternCreationWindow]);
    
    // For each active neuron, get or create connection to target neuron
    for (const an of activeNeurons) {
        const distance = an.age + 1;
        
        // Check if connection exists
        const [existing] = await this.conn.query(`
            SELECT id
            FROM connections
            WHERE from_neuron_id = ? AND to_neuron_id = ? AND distance = ?
        `, [an.neuron_id, toNeuronId, distance]);
        
        let connectionId;
        if (existing.length > 0) {
            connectionId = existing[0].id;
        } else {
            // Create new connection
            const [result] = await this.conn.query(`
                INSERT INTO connections (from_neuron_id, to_neuron_id, distance, strength)
                VALUES (?, ?, ?, 1.0)
            `, [an.neuron_id, toNeuronId, distance]);
            connectionId = result.insertId;
        }
        
        // Add to scratch table
        await this.conn.query(`
            INSERT INTO pattern_creation_future (connection_id, from_neuron_id, to_neuron_id, distance)
            VALUES (?, ?, ?, ?)
        `, [connectionId, an.neuron_id, toNeuronId, distance]);
    }
}
```

---

### Step 5: Implement `findMatchingPattern()`

```javascript
/**
 * Find existing pattern with same peak and overlapping past connections.
 * Returns pattern_neuron_id if match found, null otherwise.
 */
async findMatchingPattern(peakNeuronId) {
    
    // Find patterns with same peak
    const [candidates] = await this.conn.query(`
        SELECT pattern_neuron_id
        FROM pattern_peaks
        WHERE peak_neuron_id = ?
    `, [peakNeuronId]);
    
    if (candidates.length === 0) return null;
    
    // Calculate overlap for each candidate
    for (const candidate of candidates) {
        const [overlapResult] = await this.conn.query(`
            SELECT 
                COUNT(DISTINCT p.connection_id) as total_existing,
                COUNT(DISTINCT CASE WHEN pcp.connection_id IS NOT NULL THEN p.connection_id END) as overlap
            FROM patterns p
            LEFT JOIN pattern_creation_past pcp ON p.connection_id = pcp.connection_id
            WHERE p.pattern_neuron_id = ?
              AND p.connection_type = 'past'
        `, [candidate.pattern_neuron_id]);
        
        const { total_existing, overlap } = overlapResult[0];
        const overlapRatio = overlap / total_existing;
        
        if (overlapRatio >= this.mergePatternThreshold) {
            return candidate.pattern_neuron_id;
        }
    }
    
    return null;
}
```

---

### Step 6: Implement `mergePatternConnections()`

```javascript
/**
 * Merge future connections into existing pattern.
 * Adds new connections and reinforces existing ones.
 */
async mergePatternConnections(patternNeuronId) {
    
    // Add new future connections (or reinforce existing ones)
    await this.conn.query(`
        INSERT INTO patterns (pattern_neuron_id, connection_id, connection_type, strength)
        SELECT ?, connection_id, 'future', 1.0
        FROM pattern_creation_future
        ON DUPLICATE KEY UPDATE strength = LEAST(?, strength + 1.0)
    `, [patternNeuronId, this.maxConnectionStrength]);
}
```

---

### Step 7: Implement `createNewPatternFromScratch()`

```javascript
/**
 * Create new pattern with past and future connections from scratch tables.
 */
async createNewPatternFromScratch(peakNeuronId) {
    
    // Create pattern neuron at level 1
    const [result] = await this.conn.query('INSERT INTO neurons () VALUES ()');
    const patternNeuronId = result.insertId;
    
    // Map pattern to peak
    await this.conn.query(`
        INSERT INTO pattern_peaks (pattern_neuron_id, peak_neuron_id)
        VALUES (?, ?)
    `, [patternNeuronId, peakNeuronId]);
    
    // Insert past connections
    await this.conn.query(`
        INSERT INTO patterns (pattern_neuron_id, connection_id, connection_type, strength)
        SELECT ?, connection_id, 'past', strength
        FROM pattern_creation_past
    `, [patternNeuronId]);
    
    // Insert future connections
    await this.conn.query(`
        INSERT INTO patterns (pattern_neuron_id, connection_id, connection_type, strength)
        SELECT ?, connection_id, 'future', 1.0
        FROM pattern_creation_future
    `, [patternNeuronId]);
    
    return patternNeuronId;
}
```

---

### Step 8: Modify `inferPatterns()`

Replace hierarchical cascade with activation-based inference:

```javascript
/**
 * Pattern inference: Activate patterns based on their past connections.
 * When a pattern's past connections are all active, the pattern activates
 * and predicts its future connections.
 */
async inferPatterns() {
    
    // Clear previous pattern predictions
    await this.conn.query('TRUNCATE pattern_inferred_neurons');
    
    // Get all patterns
    const [patterns] = await this.conn.query(`
        SELECT DISTINCT pp.pattern_neuron_id, pp.peak_neuron_id
        FROM pattern_peaks pp
    `);
    
    for (const pattern of patterns) {
        
        // Check if all past connections are active
        const [activationCheck] = await this.conn.query(`
            SELECT 
                COUNT(DISTINCT p.connection_id) as total_past,
                COUNT(DISTINCT CASE WHEN ac.connection_id IS NOT NULL THEN p.connection_id END) as active_past
            FROM patterns p
            LEFT JOIN active_connections ac ON p.connection_id = ac.connection_id AND ac.level = 0
            WHERE p.pattern_neuron_id = ?
              AND p.connection_type = 'past'
        `, [pattern.pattern_neuron_id]);
        
        const { total_past, active_past } = activationCheck[0];
        const activationRatio = active_past / total_past;
        
        // Activate pattern if threshold met
        if (activationRatio >= this.patternActivationThreshold) {
            
            // Activate pattern neuron at level 1
            await this.conn.query(`
                INSERT INTO active_neurons (neuron_id, level, age)
                VALUES (?, 1, 0)
            `, [pattern.pattern_neuron_id]);
            
            // Add pattern's future connections to inference
            await this.conn.query(`
                INSERT INTO pattern_inferred_neurons (neuron_id, level, age, strength)
                SELECT c.to_neuron_id, 0, 0, SUM(p.strength)
                FROM patterns p
                JOIN connections c ON p.connection_id = c.id
                WHERE p.pattern_neuron_id = ?
                  AND p.connection_type = 'future'
                GROUP BY c.to_neuron_id
            `, [pattern.pattern_neuron_id]);
            
            if (this.debug) {
                const [predCount] = await this.conn.query(`
                    SELECT COUNT(DISTINCT neuron_id) as count
                    FROM pattern_inferred_neurons
                    WHERE level = 0
                `);
                console.log(`Pattern P${pattern.pattern_neuron_id} activated, predicted ${predCount[0].count} neurons`);
            }
        }
    }
}
```

---

### Step 9: Add Pattern Reinforcement

Modify pattern merging to handle reinforcement:

```javascript
/**
 * Reinforce pattern connections based on prediction accuracy.
 * Called during recognition phase after patterns have been matched.
 */
async reinforcePatternPredictions() {
    
    // For each active pattern at level 1
    const [activePatterns] = await this.conn.query(`
        SELECT neuron_id as pattern_neuron_id
        FROM active_neurons
        WHERE level = 1 AND age = 0
    `);
    
    for (const ap of activePatterns) {
        
        // Get pattern's future connections
        const [futureConns] = await this.conn.query(`
            SELECT p.connection_id, c.to_neuron_id, p.strength
            FROM patterns p
            JOIN connections c ON p.connection_id = c.id
            WHERE p.pattern_neuron_id = ?
              AND p.connection_type = 'future'
        `, [ap.pattern_neuron_id]);
        
        for (const fc of futureConns) {
            // Check if predicted neuron activated
            const [activated] = await this.conn.query(`
                SELECT 1
                FROM active_neurons
                WHERE neuron_id = ? AND level = 0 AND age = 0
            `, [fc.to_neuron_id]);
            
            if (activated.length > 0) {
                // Correct prediction: reinforce
                await this.conn.query(`
                    UPDATE patterns
                    SET strength = LEAST(?, strength + 1.0)
                    WHERE pattern_neuron_id = ?
                      AND connection_id = ?
                      AND connection_type = 'future'
                `, [this.maxConnectionStrength, ap.pattern_neuron_id, fc.connection_id]);
            } else {
                // Wrong prediction: weaken
                await this.conn.query(`
                    UPDATE patterns
                    SET strength = GREATEST(?, strength - ?)
                    WHERE pattern_neuron_id = ?
                      AND connection_id = ?
                      AND connection_type = 'future'
                `, [this.minConnectionStrength, this.patternNegativeReinforcement, ap.pattern_neuron_id, fc.connection_id]);
            }
        }
    }
}
```

---

### Step 10: Add Hyperparameters to `brain.js`

```javascript
// In brain.js constructor
this.patternCreationWindow = 3; // Temporal window for past connections (frames)
this.patternActivationThreshold = 1.0; // Fraction of past connections that must be active (100%)
```

---

## Testing Strategy

### Test 1: Simple Sequence Disambiguation
- Input: "ABCD" vs "ABCZ"
- Expected: Patterns created after first error, correct predictions after learning

### Test 2: Multi-Neuron Frames
- Input: "(A,B)(C,D)(E,F)(G,H)" vs "(A,B)(C,D)(E,F)(I,J)"
- Expected: Multiple patterns created, all neurons predicted correctly

### Test 3: Pattern Merging
- Input: Same sequence with slight variations
- Expected: Patterns merge, strongest predictions win

### Test 4: Long Sequences
- Input: Sequences longer than baseNeuronMaxAge
- Expected: Level 1 patterns extend context window

---

## Summary

**Implementation requires:**
1. ✅ Database schema changes (patterns table)
2. ✅ Modify negativeReinforceConnections() to detect false negatives
3. ✅ Implement createPatternFromError() and helper methods
4. ✅ Modify inferPatterns() to use activation-based logic
5. ✅ Add pattern reinforcement logic
6. ✅ Add new hyperparameters
7. ✅ Create tests to validate behavior

**Estimated effort:** 2-3 days of implementation + testing
