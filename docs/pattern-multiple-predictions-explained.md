# Pattern Multiple Predictions: Detailed Explanation

## The Question

**If we see sequences ABCD and ABCX 50% each:**
- What should the pattern be?
- What should the brain predict at C?

## The Answer

### Pattern Structure After Learning Both Sequences

**Pattern P1:**
```
Peak: C (most recent neuron before divergence)

Past Connections (context - IDENTICAL for both sequences):
  - A→B (distance=1, strength=1.0)
  - A→C (distance=2, strength=1.0)
  - B→C (distance=1, strength=1.0)

Future Connections (predictions - MERGED from both sequences):
  - A→D (distance=3, strength=50.0)  ← from ABCD sequence
  - B→D (distance=2, strength=50.0)  ← from ABCD sequence
  - C→D (distance=1, strength=50.0)  ← from ABCD sequence
  - A→X (distance=3, strength=50.0)  ← from ABCX sequence
  - B→X (distance=2, strength=50.0)  ← from ABCX sequence
  - C→X (distance=1, strength=50.0)  ← from ABCX sequence
```

### Prediction at Frame C

**Pattern P1 activates because:**
- A→B is active (A at age=2, B at age=1)
- A→C is active (A at age=2, C at age=0)
- B→C is active (B at age=1, C at age=0)
- All past connections active → pattern activates!

**Pattern P1 predicts:**
- D with total strength: 50.0 + 50.0 + 50.0 = **150.0**
- X with total strength: 50.0 + 50.0 + 50.0 = **150.0**

**Result: Both D and X predicted with equal strength (50-50)** ✓

---

## Why This Is Correct

### 1. Pattern Merging Is Essential

When we first see ABCD:
- Create pattern P1 with future connections to D

When we later see ABCX:
- Try to create pattern P2 with future connections to X
- **Pattern matching:** P2 has same peak (C) and same past connections as P1
- **Merge P2 into P1:** Add X connections to P1's future connections
- **Result:** P1 now predicts both D and X

### 2. Reinforcement Learning Balances Strengths

**Episode 1 (ABCD):**
- P1 predicts D ✓ → reinforce D connections (+1.0)
- P1 predicts X ✗ → weaken X connections (-0.1)

**Episode 2 (ABCX):**
- P1 predicts D ✗ → weaken D connections (-0.1)
- P1 predicts X ✓ → reinforce X connections (+1.0)

**After 100 episodes (50 ABCD, 50 ABCX):**
- D connections: 50 × (+1.0) + 50 × (-0.1) = 50.0 - 5.0 = 45.0
- X connections: 50 × (+1.0) + 50 × (-0.1) = 50.0 - 5.0 = 45.0

**Strengths converge to equal values!** (approximately, with some variance)

### 3. Multiple Inferred Neurons Is Correct Behavior

**The brain is saying:**
> "Given context ABC, I've learned that both D and X are equally likely outcomes. I predict both with equal confidence."

**This is NOT a failure!** It's the correct response to inherently ambiguous context.

**Accuracy interpretation:**
- If we measure "correct prediction" as "predicted neuron activated": ~50%
- If we measure "correct prediction" as "predicted all activated neurons": 100%
- The 50% accuracy reflects the inherent ambiguity, not a learning failure

---

## Comparison with Other Approaches

### Approach 1: Create Separate Patterns (NO MERGING)

**Pattern P1:** Past: ABC → Future: D
**Pattern P2:** Past: ABC → Future: X

**Problem:** Both patterns activate simultaneously!
- P1 predicts D
- P2 predicts X
- How to resolve conflict? Need additional mechanism.

**Our approach is simpler:** One pattern, multiple future connections.

### Approach 2: Only Keep Strongest Prediction

**Pattern P1:** Past: ABC → Future: D (if D more common)

**Problem:** Loses information about X!
- If distribution changes (X becomes more common), pattern can't adapt quickly
- Throws away valid learned associations

**Our approach is better:** Keep all predictions, let strengths reflect frequency.

### Approach 3: Create Higher-Level Context

**Pattern P1:** Past: ABC → Future: D
**Pattern P2:** Past: ABC → Future: X
**Level 2 pattern:** Distinguishes when to use P1 vs P2

**Problem:** Requires additional context that doesn't exist!
- If ABC is truly all the context available, level 2 can't help
- Just pushes the problem up one level

**Our approach is correct:** Accept ambiguity, predict all outcomes.

---

## Implementation Details

### Pattern Creation

```javascript
async createPatternFromError(neuronId) {
    // ... collect past connections ...
    // ... collect future connections ...
    
    // Check for existing pattern with same past connections
    const existingPatternId = await this.findMatchingPattern(peakNeuronId);
    
    if (existingPatternId) {
        // MERGE: Add future connections to existing pattern
        await this.mergePatternConnections(existingPatternId);
    } else {
        // Create new pattern
        await this.createNewPatternFromScratch(peakNeuronId);
    }
}
```

### Pattern Merging (Critical!)

```javascript
async mergePatternConnections(patternNeuronId) {
    // Add new future connections OR reinforce existing ones
    await this.conn.query(`
        INSERT INTO patterns (pattern_neuron_id, connection_id, connection_type, strength)
        SELECT ?, connection_id, 'future', 1.0
        FROM pattern_creation_future
        ON DUPLICATE KEY UPDATE strength = LEAST(?, strength + 1.0)
    `, [patternNeuronId, this.maxConnectionStrength]);
}
```

**The `ON DUPLICATE KEY UPDATE` is crucial:**
- If future connection already exists → reinforce it
- If future connection is new → add it
- This allows one pattern to accumulate multiple future connections

### Pattern Inference

```javascript
async inferPatterns() {
    for (const pattern of patterns) {
        // Check if pattern should activate
        const isActive = await this.checkPatternActivation(pattern.pattern_neuron_id);
        
        if (isActive) {
            // Add ALL future connections to inference
            await this.conn.query(`
                INSERT INTO pattern_inferred_neurons (neuron_id, level, age, strength)
                SELECT c.to_neuron_id, 0, 0, SUM(p.strength)
                FROM patterns p
                JOIN connections c ON p.connection_id = c.id
                WHERE p.pattern_neuron_id = ?
                  AND p.connection_type = 'future'
                GROUP BY c.to_neuron_id
            `, [pattern.pattern_neuron_id]);
        }
    }
}
```

**Note:** `GROUP BY c.to_neuron_id` sums strengths of all connections to the same neuron.
- If 3 connections to D (from A, B, C) each with strength 50.0
- Total prediction for D: 150.0

---

## Multi-Neuron Frames

**This design naturally handles multi-neuron frames!**

### Example: Frame with 2 neurons

**Sequence:** (A,B) → (C,D) → (E,F)

**Pattern P1 for E:**
- Past: A→C, A→D, B→C, B→D
- Future: A→E, B→E, C→E, D→E

**Pattern P2 for F:**
- Past: A→C, A→D, B→C, B→D (SAME as P1!)
- Future: A→F, B→F, C→F, D→F

**Pattern matching:** P2 has same past as P1, but different peak (E vs F)

**Should they merge?** NO! Different peaks mean different patterns.

**Result:** Both P1 and P2 activate, predict E and F respectively.

### Example: Ambiguous multi-neuron frame

**Sequence A:** (A,B) → (C,D) → (E,F)
**Sequence B:** (A,B) → (C,D) → (G,H)

**Pattern P1 (merged):**
- Past: A→C, A→D, B→C, B→D
- Future: A→E, B→E, C→E, D→E (strength=50) + A→G, B→G, C→G, D→G (strength=50)

**Pattern P2 (merged):**
- Past: A→C, A→D, B→C, B→D
- Future: A→F, B→F, C→F, D→F (strength=50) + A→H, B→H, C→H, D→H (strength=50)

**At frame (C,D), patterns predict:**
- E: 200 (from P1)
- F: 200 (from P2)
- G: 200 (from P1)
- H: 200 (from P2)

**All four neurons predicted with equal strength!** ✓

---

## Summary

### Key Points

1. ✅ **Patterns MUST merge future connections when past connections are identical**
2. ✅ **One pattern can predict multiple neurons simultaneously**
3. ✅ **Prediction strengths reflect learned frequencies**
4. ✅ **Ambiguous context → multiple predictions (correct behavior)**
5. ✅ **Multi-neuron frames handled naturally**

### Database Schema

**Patterns table supports multiple future connections:**
```sql
CREATE TABLE patterns (
    pattern_neuron_id BIGINT UNSIGNED NOT NULL,
    connection_id BIGINT UNSIGNED NOT NULL,
    connection_type ENUM('past', 'future') NOT NULL,
    strength DOUBLE NOT NULL DEFAULT 1.0,
    PRIMARY KEY (pattern_neuron_id, connection_id, connection_type)
)
```

**One pattern_neuron_id can have:**
- Multiple past connections (all must be active to trigger)
- Multiple future connections (all are predicted when triggered)

### Expected Behavior

**For ABCD vs ABCX (50% each):**
- ✅ One pattern created (merged)
- ✅ Pattern has 3 past connections (A→B, A→C, B→C)
- ✅ Pattern has 6 future connections (3 to D, 3 to X)
- ✅ At frame C, predicts both D and X with equal strength
- ✅ Accuracy ~50% (correct for ambiguous context)

**For ABCD vs ABCX (80% vs 20%):**
- ✅ Same pattern structure
- ✅ D connections stronger (strength ≈ 80)
- ✅ X connections weaker (strength ≈ 20)
- ✅ At frame C, predicts D stronger than X
- ✅ Accuracy ~80% (matches distribution)

**The design is correct and complete!**
