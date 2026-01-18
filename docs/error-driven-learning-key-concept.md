# Error-Driven Learning - Key Concept

## The Core Change

**ONLY ONE CHANGE:** Remove pattern creation from recognition, add it to error-driven learning phase.

```
OLD: Pattern created during recognition (every frame)
NEW: Pattern created only when confident prediction fails
```

## Who Is Learning?

**Critical Understanding:** The neuron doing the predicting is the one that needs to learn.

### Example Scenario

```
Frame N-2:
  Neuron X (active, age=0)

Frame N-1:
  Neuron A (active, age=0)
  Neuron X (active, age=1)

Frame N:
  Neuron B (active, age=0)
  Neuron A (active, age=1)
  Neuron X (active, age=2)

  Neuron C (active, age=0)
  Neuron B (active, age=1)
  Neuron A (active, age=2)
  Neuron X (active, age=3)

  Connection: C → D (distance=1, strength=0.8)

  Prediction: Neuron D will appear in next frame (strength: 0.8)
  Predictor: Neuron C (the one with the connection to D)

Frame N+1:
  Actual: Neuron E appears (not D)

  Result: Prediction FAILED

  Question: Who needs to learn from this error?
  Answer: Neuron C (the predictor)
```

## Pattern Structure for Error

```
Error Pattern Created at Level 1:

  Peak Neuron: C (the predictor, not D the failed prediction)

  pattern_past (for recognition):
    - ALL connections TO C at ALL distances (the context when C was active)
    - Example:
      - B→C (distance=1) - immediate predecessor
      - A→C (distance=2) - 2 frames back
      - X→C (distance=3) - 3 frames back
      - ... up to distance=9 (contextLength)
    - Purpose: Recognize when C appears in this SPECIFIC context again
    - Captures up to 9 frames of temporal history!

  pattern_future (for inference):
    - ALL connections FROM C (what C predicts)
    - Example: C→D (distance=1), C→E (distance=2), etc.
    - Purpose: Provide top-down predictions when pattern is active

  pattern_peaks mapping:
    - pattern_neuron_id → C (the predictor)
```

**Critical Insight:** Multiple-distance connections capture LONG temporal context!
- Not just immediate predecessor (distance=1)
- But also what happened 2, 3, 4... up to 9 frames ago
- This enables rich context differentiation

## How Pattern Is Used

### Recognition Phase (Future Frames)

```
When neuron C appears again:
  Active neurons: Y (age=3), A (age=2), B (age=1), C (age=0)

  1. Check pattern_past: Do incoming connections match at correct distances?
     - B→C at distance=1? Check if B is at age=1 ✓
     - A→C at distance=2? Check if A is at age=2 ✓
     - X→C at distance=3? Check if X is at age=3 ✗ (Y is there instead)

  2. If enough connections match: Activate pattern neuron at level+1
  3. Reinforce matched pattern connections (Hebbian learning)

Pattern represents: "C in this specific temporal context"
```

**Key:** Different active neurons at different distances = different contexts!

### Inference Phase

```
When pattern neuron is active at level+1:
  1. Get pattern_future connections
  2. Unpack predictions from C (weighted by pattern strength)
  3. Provide top-down context to lower levels

Pattern predicts: "What C predicts when in this specific context"
```

## Why This Makes Sense

### Biological Analogy

A neuron learns from its own prediction errors:
- Neuron C fired and predicted D would fire next
- D didn't fire
- C needs to adjust its predictions (synaptic weights)
- Pattern captures C's prediction behavior in specific contexts

### Learning Dynamics

**Without error-driven learning:**
- Patterns created for every peak
- Many patterns, including noise
- Hard to distinguish important patterns

**With error-driven learning:**
- Patterns created only when confident predictions fail
- Fewer patterns, focused on errors
- Pattern represents: "This neuron made a mistake in this context"
- Future occurrences: Pattern helps refine predictions

## Initial Learning Phase

**Question:** How does the system learn initially when there are no patterns?

**Answer:** Connection inference only.

```
Frames 1-N: No patterns exist
  - Only connection inference runs
  - Connections created via Hebbian learning
  - Predictions made from connections
  - Some predictions fail
  
Frame N+1: First error with high confidence
  - validateAndLearnFromErrors() runs
  - First pattern created for predictor neuron
  - Pattern captures error context
  
Frames N+2+: Patterns start to accumulate
  - Patterns created only on confident errors
  - Patterns used in recognition and inference
  - System builds hierarchical representations
```

## Pattern Peak: Predictor vs Predicted

### WRONG Understanding ❌

```
Neuron C predicts Neuron D (fails)
Pattern peak = D (the failed prediction)

Problem: D didn't appear, so pattern can never activate!
```

### CORRECT Understanding ✅

```
Neuron C predicts Neuron D (fails)
Pattern peak = C (the predictor)

When C appears again with similar context:
  - Pattern activates (C is present)
  - Pattern provides learned predictions
  - Pattern strength reflects C's reliability in this context
```

## Multiple Patterns from One Error

### Case 1: Multiple Predictors for Same Failed Prediction

If multiple neurons predicted the same failed neuron:

```
Neuron A predicts D (fails)
Neuron B predicts D (fails)
Neuron C predicts D (fails)

Result: Create 3 separate patterns
  - Pattern_1: peak = A, pattern_past = {connections TO A at distances 1-9}
  - Pattern_2: peak = B, pattern_past = {connections TO B at distances 1-9}
  - Pattern_3: peak = C, pattern_past = {connections TO C at distances 1-9}

Each neuron learns independently from its own error.
Each pattern captures different temporal context (what led to A vs B vs C being active).
```

### Case 2: Same Neuron in Different Contexts

The same neuron can be peak of MULTIPLE patterns:

```
Context 1: X → A → B → C (predicts D, but E appears)
  - Pattern_1: peak = C, pattern_past includes X→C at distance=3

Context 2: Y → A → B → C (predicts D, but F appears)
  - Pattern_2: peak = C, pattern_past includes Y→C at distance=3

Both patterns have peak = C, but different pattern_past!
Pattern_1 activates when X is at distance=3
Pattern_2 activates when Y is at distance=3
```

**This is how the brain differentiates contexts!**

## Pattern Has Single Peak

**Rule:** One pattern = one peak neuron

**Why?**
- Biologically: One neuron is the learner
- Computationally: Easier to merge/split patterns
- Conceptually: Clear ownership of learning

**Can a pattern have multiple peaks?** NO
- Each pattern belongs to one neuron
- That neuron is learning its prediction behavior

**Can a pattern predict multiple neurons?** YES
- pattern_future contains all connections FROM the peak
- Peak neuron may predict multiple future neurons
- All predictions are part of the pattern

## Hebbian Learning Preserved

**Connection Level (Horizontal):**
```
Neurons fire together → Connection strengthened
reinforceConnections() in activateNeurons()
NO CHANGE
```

**Pattern Level (Vertical):**
```
Pattern matches observed context → Pattern strengthened
mergeMatchedPatterns() in recognizeLevelPatterns()
NO CHANGE
```

**What Changes:**
```
Pattern creation moved from recognition to error-driven learning
createNewPatterns() removed from recognizeLevelPatterns()
createErrorPatterns() added to validateAndLearnFromErrors()
```

## Summary

1. **Pattern peak = predictor neuron** (who made the error)
2. **Pattern past = context** (ALL connections TO predictor at distances 1-9)
3. **Pattern future = predictions** (ALL connections FROM predictor)
4. **Multiple-distance connections** capture up to 9 frames of temporal history
5. **Pattern activates** when predictor appears with matching context (neurons at correct distances)
6. **Pattern learns** via Hebbian reinforcement when activated
7. **Pattern created** only when confident prediction fails
8. **Multiple patterns per neuron** for different contexts (differentiated by pattern_past)
9. **Context differentiation** works because different histories have different active neurons at different distances
10. **Initial learning** uses connection inference only
11. **Hebbian learning** preserved for both connections and patterns
12. **Only change** is when/how patterns are created
13. **Will memorize** any sequence with learnable temporal structure
14. **Won't memorize** truly random/unpredictable sequences (correct behavior)

