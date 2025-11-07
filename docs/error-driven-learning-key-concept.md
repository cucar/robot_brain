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
Frame N:
  Neuron A (active, age=0)
  Neuron B (active, age=0)
  Neuron C (active, age=0)
  
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
    - Connections TO C (the context when C was active)
    - Example: A→C, B→C
    - Purpose: Recognize when C appears in this context again
  
  pattern_future (for inference):
    - Connections FROM C (what C predicts)
    - Example: C→D (the failed prediction), C→E, etc.
    - Purpose: Provide top-down predictions when pattern is active
  
  pattern_peaks mapping:
    - pattern_neuron_id → C (the predictor)
```

## How Pattern Is Used

### Recognition Phase (Future Frames)

```
When neuron C appears again:
  1. Check pattern_past: Do incoming connections match?
  2. If yes: Activate pattern neuron at level+1
  3. Reinforce pattern connections (Hebbian learning)
  
Pattern represents: "C in this specific context"
```

### Inference Phase

```
When pattern neuron is active at level+1:
  1. Get pattern_future connections
  2. Unpack predictions from C
  3. Provide top-down context to lower levels
  
Pattern predicts: "What C predicts when in this context"
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

If multiple neurons predicted the same failed neuron:

```
Neuron A predicts D (fails)
Neuron B predicts D (fails)
Neuron C predicts D (fails)

Result: Create 3 separate patterns
  - Pattern_1: peak = A
  - Pattern_2: peak = B
  - Pattern_3: peak = C

Each neuron learns independently from its own error.
```

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
mergeMatchedPatterns() in activateLevelPatterns()
NO CHANGE
```

**What Changes:**
```
Pattern creation moved from recognition to error-driven learning
createNewPatterns() removed from activateLevelPatterns()
createErrorPatterns() added to validateAndLearnFromErrors()
```

## Summary

1. **Pattern peak = predictor neuron** (who made the error)
2. **Pattern past = context** (connections TO predictor)
3. **Pattern future = predictions** (connections FROM predictor)
4. **Pattern activates** when predictor appears with matching context
5. **Pattern learns** via Hebbian reinforcement when activated
6. **Pattern created** only when confident prediction fails
7. **One pattern per predictor** for each error context
8. **Initial learning** uses connection inference only
9. **Hebbian learning** preserved for both connections and patterns
10. **Only change** is when/how patterns are created

