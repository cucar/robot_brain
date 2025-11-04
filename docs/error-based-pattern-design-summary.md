# Error-Based Pattern Creation: Design Summary

## Core Concept

**Patterns are created ONLY when predictions fail (surprise-based learning).**

This is biologically inspired by:
- Prediction error signals (dopamine)
- Hippocampal memory formation (novelty/surprise)
- Free energy principle (minimize surprise)
- Memory consolidation (useful memories persist, useless ones fade)

---

## Pattern Structure: Past vs Future Connections

### Past Connections (Context)
- Define WHEN the pattern should activate
- All connections that were active in the temporal window before the error
- Typically 3-frame window (ages 1-3): `patternCreationWindow = 3`
- These connections represent the "context" that led to the prediction error

### Future Connections (Prediction)
- Define WHAT the pattern should predict
- All connections to the neuron that failed to be predicted
- These connections represent the "correction" to the prediction error

### Example
When neuron F fails to be predicted after sequence "ABC":
- **Past connections:** All active connections to A, B, C
- **Future connections:** All connections to F from active neurons
- **Peak neuron:** C (most recent neuron, age=1)

---

## Pattern Creation Algorithm

### When Error Occurs (False Negative)

**Input:** Neuron N activated but was not predicted

**Step 1: Determine peak neuron**
- Most recent active neuron (age=1)
- This is the "decision point" where context matters

**Step 2: Collect past connections (context)**
```sql
SELECT c.id, c.from_neuron_id, c.to_neuron_id, c.distance, c.strength
FROM connections c
JOIN active_connections ac ON c.id = ac.connection_id
WHERE ac.level = 0
  AND ac.age BETWEEN 1 AND patternCreationWindow
```

**Step 3: Collect future connections (prediction)**
```sql
-- Get all connections to the unpredicted neuron from active neurons
SELECT c.id, c.from_neuron_id, c.to_neuron_id, c.distance
FROM active_neurons an
WHERE an.level = 0
  AND an.age <= patternCreationWindow
-- Create connections if they don't exist
-- from_neuron_id = an.neuron_id
-- to_neuron_id = N
-- distance = an.age + 1
```

**Step 4: Create pattern**
```sql
-- Create pattern neuron at level 1
INSERT INTO neurons () VALUES ();
SET @pattern_neuron_id = LAST_INSERT_ID();

-- Map pattern to peak
INSERT INTO pattern_peaks (pattern_neuron_id, peak_neuron_id)
VALUES (@pattern_neuron_id, @peak_neuron_id);

-- Store past connections
INSERT INTO patterns (pattern_neuron_id, connection_id, connection_type, strength)
SELECT @pattern_neuron_id, connection_id, 'past', 1.0
FROM past_connections;

-- Store future connections
INSERT INTO patterns (pattern_neuron_id, connection_id, connection_type, strength)
SELECT @pattern_neuron_id, connection_id, 'future', 1.0
FROM future_connections;
```

**Step 5: Create base-level connections**
- Also create the future connections at level 0 (if they don't exist)
- This ensures connection inference can use them in future frames

---

## Pattern Activation Algorithm

### During Inference Phase

**For each existing pattern:**

**Step 1: Check if past connections are active**
```sql
SELECT COUNT(*) as active_count
FROM patterns p
JOIN connections c ON p.connection_id = c.id
JOIN active_connections ac ON c.id = ac.connection_id
WHERE p.pattern_neuron_id = @pattern_id
  AND p.connection_type = 'past'
  AND ac.level = 0
```

**Step 2: Calculate activation strength**
```sql
SELECT COUNT(*) as total_past, 
       SUM(CASE WHEN ac.connection_id IS NOT NULL THEN 1 ELSE 0 END) as active_past
FROM patterns p
LEFT JOIN active_connections ac ON p.connection_id = ac.connection_id AND ac.level = 0
WHERE p.pattern_neuron_id = @pattern_id
  AND p.connection_type = 'past'
```

**Activation threshold:** 
- Require 100% of past connections to be active? (strict)
- Or require >= 66% of past connections to be active? (lenient)
- **Recommendation: Start with 100%, relax if needed**

**Step 3: If activated, predict future connections**
```sql
-- Add pattern's future connections to inference
INSERT INTO inferred_connections (level, connection_id, from_neuron_id, to_neuron_id, strength)
SELECT 0, p.connection_id, c.from_neuron_id, c.to_neuron_id, p.strength
FROM patterns p
JOIN connections c ON p.connection_id = c.id
WHERE p.pattern_neuron_id = @pattern_id
  AND p.connection_type = 'future'
```

**Step 4: Activate pattern neuron at level 1**
```sql
INSERT INTO active_neurons (neuron_id, level, age)
VALUES (@pattern_neuron_id, 1, 0)
```

---

## Pattern Matching and Merging

### When Creating New Pattern

**Step 1: Find existing patterns with same peak**
```sql
SELECT pp.pattern_neuron_id
FROM pattern_peaks pp
WHERE pp.peak_neuron_id = @new_peak_neuron_id
```

**Step 2: Calculate overlap of past connections**
```sql
SELECT existing.pattern_neuron_id,
       COUNT(DISTINCT CASE WHEN new.connection_id IS NOT NULL THEN existing.connection_id END) as overlap,
       COUNT(DISTINCT existing.connection_id) as total_existing
FROM patterns existing
LEFT JOIN new_pattern_past_connections new ON existing.connection_id = new.connection_id
WHERE existing.pattern_neuron_id IN (candidate_patterns)
  AND existing.connection_type = 'past'
GROUP BY existing.pattern_neuron_id
HAVING overlap / total_existing >= mergePatternThreshold
```

**Step 3: If match found, merge**
```sql
-- Add new future connections to existing pattern
INSERT INTO patterns (pattern_neuron_id, connection_id, connection_type, strength)
SELECT @existing_pattern_id, connection_id, 'future', 1.0
FROM new_pattern_future_connections
ON DUPLICATE KEY UPDATE strength = strength + 1.0;

-- Reinforce existing future connections that were also in new pattern
UPDATE patterns p
JOIN new_pattern_future_connections nf ON p.connection_id = nf.connection_id
SET p.strength = p.strength + 1.0
WHERE p.pattern_neuron_id = @existing_pattern_id
  AND p.connection_type = 'future';
```

**Step 4: If no match, create new pattern**
- Follow pattern creation algorithm above

---

## Pattern Learning (Reinforcement)

### Positive Reinforcement (Correct Predictions)

When pattern's future connection predicts correctly:
```sql
UPDATE patterns
SET strength = LEAST(maxConnectionStrength, strength + 1.0)
WHERE pattern_neuron_id = @pattern_id
  AND connection_id = @correct_connection_id
  AND connection_type = 'future'
```

### Negative Reinforcement (Wrong Predictions)

When pattern's future connection predicts incorrectly:
```sql
UPDATE patterns
SET strength = GREATEST(minConnectionStrength, strength - patternNegativeReinforcement)
WHERE pattern_neuron_id = @pattern_id
  AND connection_id = @wrong_connection_id
  AND connection_type = 'future'
```

### Forget Cycle

Every N frames, decay all pattern connections:
```sql
UPDATE patterns
SET strength = GREATEST(minConnectionStrength, strength - patternForgetRate)
WHERE connection_type IN ('past', 'future')
```

Delete weak connections:
```sql
DELETE FROM patterns
WHERE strength <= minConnectionStrength
```

Delete patterns with no connections:
```sql
DELETE pp FROM pattern_peaks pp
LEFT JOIN patterns p ON pp.pattern_neuron_id = p.pattern_neuron_id
WHERE p.pattern_neuron_id IS NULL
```

---

## Integration with Existing System

### Changes to `negativeReinforceConnections()`

**Current:** Only weakens connections that predicted incorrectly

**New:** Also triggers pattern creation for false negatives

```javascript
async negativeReinforceConnections() {
    // Find false positives (predicted but didn't activate)
    const [falsePositives] = await this.conn.query(`...`);
    
    // Apply negative reinforcement to false positives
    await this.conn.query(`UPDATE connections SET strength = ...`);
    
    // NEW: Find false negatives (activated but weren't predicted)
    const [falseNegatives] = await this.conn.query(`
        SELECT an.neuron_id, an.level
        FROM active_neurons an
        WHERE an.age = 0
          AND NOT EXISTS (
              SELECT 1 FROM connection_inferred_neurons cin
              WHERE cin.neuron_id = an.neuron_id AND cin.level = an.level
          )
          AND NOT EXISTS (
              SELECT 1 FROM pattern_inferred_neurons pin
              WHERE pin.neuron_id = an.neuron_id AND pin.level = an.level
          )
    `);
    
    // NEW: Create patterns for false negatives
    for (const fn of falseNegatives) {
        await this.createPatternFromError(fn.neuron_id, fn.level);
    }
}
```

### New Method: `createPatternFromError()`

```javascript
async createPatternFromError(neuronId, level) {
    // Only create patterns at level 0 for now
    if (level !== 0) return;
    
    // Step 1: Determine peak neuron (most recent active neuron)
    const peakNeuronId = await this.getPeakNeuronForError(level);
    
    // Step 2: Collect past connections
    const pastConnections = await this.getActiveConnectionsInWindow(level);
    
    // Step 3: Collect/create future connections
    const futureConnections = await this.getOrCreateConnectionsToNeuron(neuronId, level);
    
    // Step 4: Check for existing patterns to merge with
    const existingPattern = await this.findMatchingPattern(peakNeuronId, pastConnections);
    
    if (existingPattern) {
        // Merge: add future connections to existing pattern
        await this.mergePatternConnections(existingPattern, futureConnections);
    } else {
        // Create new pattern
        await this.createNewPattern(peakNeuronId, pastConnections, futureConnections);
    }
}
```

### Changes to `inferPatterns()`

**Current:** Cascades predictions from higher levels down

**New:** Activates patterns based on past connections

```javascript
async inferPatterns() {
    // Get all patterns
    const [patterns] = await this.conn.query(`
        SELECT DISTINCT pattern_neuron_id, peak_neuron_id
        FROM pattern_peaks
    `);
    
    for (const pattern of patterns) {
        // Check if pattern's past connections are active
        const isActive = await this.checkPatternActivation(pattern.pattern_neuron_id);
        
        if (isActive) {
            // Activate pattern neuron at level 1
            await this.conn.query(`
                INSERT INTO active_neurons (neuron_id, level, age)
                VALUES (?, 1, 0)
            `, [pattern.pattern_neuron_id]);
            
            // Add pattern's future connections to inference
            await this.addPatternPredictions(pattern.pattern_neuron_id);
        }
    }
}
```

---

## Database Schema Changes

### Modify `patterns` table

```sql
ALTER TABLE patterns
ADD COLUMN connection_type ENUM('past', 'future') NOT NULL AFTER connection_id,
DROP PRIMARY KEY,
ADD PRIMARY KEY (pattern_neuron_id, connection_id, connection_type),
ADD INDEX idx_pattern_type (pattern_neuron_id, connection_type);
```

---

## Hyperparameters

- `patternCreationWindow = 3`: Temporal window for past connections (frames)
- `patternActivationThreshold = 1.0`: Fraction of past connections that must be active (100%)
- `mergePatternThreshold = 0.66`: Overlap threshold for pattern merging (66%)
- `patternNegativeReinforcement = 0.1`: Strength decrease for wrong predictions
- `patternForgetRate = 1.0`: Strength decrease per forget cycle

---

## Expected Behavior

### Single Sequence (e.g., "ABCD" repeated)
- Episode 1: Learn base connections
- Episode 2-3: Connections reinforced, no errors, no patterns
- Result: Pure connection-based inference, 100% accuracy

### Two Sequences with Shared Prefix (e.g., "ABCD" vs "ABCZ")
- Episode 1: Learn "ABCD"
- Episode 2: See "ABCZ", error on Z, create pattern P1 (predicts Z after ABC)
- Episode 3: See "ABCD", error on D, create pattern P2 (predicts D after ABC)
- Pattern matching: P1 and P2 have same past connections (ABC context)
- Pattern merging: Merge into single pattern that predicts both D and Z
- Episodes 4+: Pattern makes both predictions, correct one gets reinforced
- Result: If sequences equally common, ~50% accuracy (correct for ambiguous context)
- Result: If one sequence more common, that prediction dominates

### Long Sequences (beyond baseNeuronMaxAge)
- Level 0 patterns capture 3-frame context
- Level 1 patterns use level 0 pattern activations as context
- This extends effective context window exponentially
- Result: Can learn sequences longer than baseNeuronMaxAge

---

## Conclusion

**Error-based pattern creation provides:**
1. ✅ Biologically plausible learning (surprise-based)
2. ✅ Automatic context discovery (past connections)
3. ✅ Efficient pattern creation (only when needed)
4. ✅ Graceful ambiguity handling (pattern competition)
5. ✅ Self-regulating system (forget cycle)
6. ✅ Scalable to multi-neuron frames
7. ✅ Extends temporal context hierarchically
