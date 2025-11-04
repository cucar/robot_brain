# Error-Based Pattern Quick Reference

## Key Concepts

### Pattern Structure

**Pattern = Context + Prediction**

- **Past Connections** (stored in `patterns_past`): Define the context that activates the pattern
- **Future Connections** (stored in `patterns_future`): Define what the pattern predicts
- **Peak Neuron** (stored in `pattern_peaks`): The "decision point" - most recent neuron before error

### Pattern Lifecycle

1. **Creation**: When prediction fails (false negative)
2. **Activation**: When ALL past connections are active
3. **Prediction**: Activated pattern predicts its future connections
4. **Reinforcement**: Correct predictions strengthened, incorrect weakened
5. **Forgetting**: Weak connections deleted

---

## Database Tables

### Core Pattern Tables

```sql
patterns_past (pattern_neuron_id, connection_id, strength)
patterns_future (pattern_neuron_id, connection_id, strength)
pattern_peaks (pattern_neuron_id, peak_neuron_id)
```

### Scratch Tables (MEMORY)

```sql
-- Error-based pattern creation
pattern_creation_past (connection_id)
pattern_creation_future (connection_id)

-- Pattern inference
activated_patterns (pattern_neuron_id, activation_strength)
pattern_predicted_neurons (neuron_id, total_strength)

-- Existing (used by both systems)
observed_patterns (peak_neuron_id, connection_id)
observed_peaks (peak_neuron_id, total_strength, connection_count)
matched_patterns (peak_neuron_id, pattern_neuron_id)
matched_peaks (peak_neuron_id)
```

---

## Key Algorithms

### Pattern Creation from Errors

```
1. Find false negatives (neurons activated but not predicted)
2. Collect past connections (ALL active connections, ages 1-baseNeuronMaxAge)
3. Collect future connections (connections to false negative neurons)
4. Find peak neuron (most recent active neuron, age=1)
5. Populate observed_patterns with past connections
6. Match to existing patterns (based on past connections)
7. Merge or create new pattern
8. Add future connections to pattern
```

### Pattern Activation & Inference

```
1. For each pattern:
   - Check if ALL past connections are active
   - If yes, pattern activates
2. For each activated pattern:
   - Predict all future connections
   - Aggregate by to_neuron_id (sum strengths)
3. Filter predictions:
   - If 1 neuron: keep it
   - If multiple: keep only above-average strength
```

### Pattern Matching

```
1. For each known pattern:
   - Count how many past connections match observed connections
   - If overlap >= mergePatternThreshold (50%): MATCH
2. Matched patterns get reinforced
3. Unmatched peaks get new patterns created
```

---

## Example Walkthrough

### Sequence: ABCD (50%) vs ABCX (50%)

**Episode 1: ABCD**

Frame 1: A activates
Frame 2: B activates
Frame 3: C activates
Frame 4: D activates (not predicted - ERROR!)

**Pattern Creation:**
- Past: A→B, A→C, B→C (all active connections)
- Future: A→D, B→D, C→D (connections to D)
- Peak: C (age=1 when error occurred)
- Create Pattern P1

**Episode 2: ABCX**

Frame 1: A activates
Frame 2: B activates
Frame 3: C activates
- P1 activates (past connections match!)
- P1 predicts D
Frame 4: X activates (not D - ERROR!)

**Pattern Merging:**
- Try to create new pattern for X
- Past: A→B, A→C, B→C (same as P1!)
- Match found: P1
- Merge: Add future connections to X
- P1 now has:
  - Past: A→B, A→C, B→C
  - Future: A→D, B→D, C→D, A→X, B→X, C→X

**Episode 3: ABCD**

Frame 3: C activates
- P1 activates
- P1 predicts: D (strength=50×3=150), X (strength=50×3=150)
- Both above average → both predicted
Frame 4: D activates
- D was predicted ✓ → reinforce D connections (+1.0)
- X was predicted ✗ → weaken X connections (-0.1)

**After 100 episodes (50 ABCD, 50 ABCX):**
- D connections: ~50 strength
- X connections: ~50 strength
- P1 predicts both D and X with equal strength ✓

---

## Hyperparameters

```javascript
// Existing
baseNeuronMaxAge = 5              // How long neurons stay active
mergePatternThreshold = 0.50      // 50% overlap for pattern matching
patternNegativeReinforcement = 0.1 // How much to weaken unobserved connections
minConnectionStrength = 0.0       // Delete connections below this
maxConnectionStrength = 1000.0    // Clamp connections above this

// New
patternActivationThreshold = 1.0  // 100% of past connections must be active
```

---

## Key Differences from Old System

### Old Pattern System (Hierarchical)

- **Purpose**: Hierarchical abstraction (level N → level N-1)
- **Structure**: Pattern neuron → peak neuron
- **Activation**: Pattern neuron activated → predict peak neuron at level below
- **Connections**: Only past connections (to peak)
- **Inference**: Cascade down levels

### New Pattern System (Error-Based)

- **Purpose**: Extend temporal context beyond baseNeuronMaxAge
- **Structure**: Past connections (context) + Future connections (predictions)
- **Activation**: Past connections active → predict future connections
- **Connections**: Both past AND future
- **Inference**: Activate on past, predict future, filter by strength

### Integration

Both systems coexist:
- **Hierarchical patterns**: `activatePatternNeurons()` - still used for multi-level abstraction
- **Error-based patterns**: `createPatternsFromErrors()` + new `inferPatterns()` - extends temporal reach

---

## Common Pitfalls

### ❌ Don't match on future connections
Pattern matching uses ONLY past connections. Future connections are predictions, not context.

### ❌ Don't require peak neuron to be predicted
Peak neuron is the decision point, not necessarily what gets predicted. Future connections define predictions.

### ❌ Don't filter predictions too aggressively
If only 1 neuron predicted, keep it regardless of strength. Only filter when multiple predictions compete.

### ❌ Don't forget to aggregate by to_neuron_id
Multiple connections can point to same neuron. Sum their strengths before filtering.

### ❌ Don't create patterns for every frame
Only create patterns when predictions FAIL (false negatives). This is surprise-based learning.

---

## Testing Checklist

### Pattern Creation
- [ ] False negatives detected correctly
- [ ] Past connections include ALL active connections (ages 1-baseNeuronMaxAge)
- [ ] Future connections include ALL connections to false negatives
- [ ] Peak neuron is most recent (age=1)
- [ ] Pattern matching works (50% overlap threshold)
- [ ] Pattern merging adds future connections
- [ ] New patterns created when no match

### Pattern Inference
- [ ] Patterns activate when ALL past connections active
- [ ] Patterns don't activate when some past connections missing
- [ ] Future connections predicted correctly
- [ ] Aggregation by to_neuron_id works
- [ ] Single prediction kept regardless of strength
- [ ] Multiple predictions filtered by average
- [ ] Above-average predictions kept

### Pattern Learning
- [ ] Correct predictions reinforced (+1.0)
- [ ] Incorrect predictions weakened (-0.1)
- [ ] Strengths converge to frequency distribution
- [ ] Ambiguous context predicts multiple neurons
- [ ] Weak connections deleted by forget cycle

### Integration
- [ ] Error-based patterns created after negative reinforcement
- [ ] Pattern inference runs before conflict resolution
- [ ] Both hierarchical and error-based patterns coexist
- [ ] No interference between systems

---

## Debug Output

### Pattern Creation
```
Creating patterns from prediction errors
Found 2 prediction errors
Activated 0 patterns
Creating 1 new patterns
```

### Pattern Inference
```
Inferring patterns
Activated 3 patterns
Patterns predicted 5 neurons (before filtering)
Kept 2 pattern predictions (above average)
```

### Pattern Matching
```
Matching observed patterns to known patterns
Matched 1 pattern-peak pairs
Merging matched patterns
```

---

## Performance Considerations

### Indexes Required

```sql
-- patterns_past
INDEX idx_connection (connection_id)
INDEX idx_pattern_strength (pattern_neuron_id, strength)

-- patterns_future
INDEX idx_connection (connection_id)
INDEX idx_pattern_strength (pattern_neuron_id, strength)

-- pattern_peaks
INDEX idx_peak (peak_neuron_id)

-- active_connections
INDEX idx_level_age (level, age)
INDEX idx_connection (connection_id)
```

### Query Optimization

- Use `TRUNCATE` instead of `DELETE` for scratch tables
- Use `INSERT IGNORE` to avoid duplicate key errors
- Use `ON DUPLICATE KEY UPDATE` for upserts
- Use `CROSS JOIN` for Cartesian products (pattern × future connections)
- Use `LEFT JOIN` for optional matches
- Use `EXISTS` for existence checks (faster than `IN`)

### Memory Usage

All pattern tables use `ENGINE=MEMORY` for speed. Monitor memory usage:
- `patterns_past`: ~1M patterns × 10 connections = 10M rows
- `patterns_future`: ~1M patterns × 10 connections = 10M rows
- Total: ~20M rows × 24 bytes = ~480 MB

If memory is constrained, consider:
- Reducing `maxConnectionStrength` (fewer strong patterns)
- Increasing `patternNegativeReinforcement` (faster forgetting)
- Running forget cycle more frequently

---

## Migration Path

### Phase 1: Database Changes
1. Create new tables (`patterns_past`, `patterns_future`, scratch tables)
2. Keep old `patterns` table temporarily
3. Test schema changes

### Phase 2: Method Updates
1. Update `matchObservedPatterns()` to use `patterns_past`
2. Update `mergeMatchedPatterns()` to handle both tables
3. Update `createNewPatterns()` to populate both tables
4. Test with existing functionality

### Phase 3: New Inference
1. Implement new `inferPatterns()` methods
2. Test pattern activation and prediction
3. Test filtering logic

### Phase 4: Error-Based Creation
1. Implement `createPatternsFromErrors()` methods
2. Test pattern creation from errors
3. Test pattern merging

### Phase 5: Integration
1. Add calls to `processFrame()`
2. Test end-to-end
3. Drop old `patterns` table

### Phase 6: Optimization
1. Monitor performance
2. Tune hyperparameters
3. Add indexes as needed

---

## Success Metrics

### Pattern Creation
- Patterns created only when predictions fail ✓
- One pattern per error event (not per neuron) ✓
- Pattern merging prevents duplicates ✓

### Pattern Inference
- Patterns activate based on context ✓
- Predictions extend beyond baseNeuronMaxAge ✓
- Ambiguous context predicts multiple outcomes ✓

### Pattern Learning
- Prediction accuracy improves over time ✓
- Strengths reflect frequency distribution ✓
- Unused patterns decay and disappear ✓

### Performance
- Pattern creation: < 10ms per frame ✓
- Pattern inference: < 10ms per frame ✓
- Memory usage: < 500 MB ✓

**Ready to implement!** 🚀

