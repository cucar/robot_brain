# Error-Driven Learning Architecture - Flow Diagrams

## Current Architecture vs New Architecture

### Current Architecture Flow

```
processFrame(frame, globalReward)
│
├─► applyRewards(globalReward)
│   └─► Strengthen/weaken connections based on reward
│
├─► ageNeurons()
│   └─► Age all context by 1 frame
│
├─► executeOutputs()
│   └─► Execute age=1 predictions
│
├─► recognizeNeurons(frame)
│   ├─► getFrameNeurons() - Find/create base neurons
│   ├─► activateNeurons() - Activate at level 0
│   │   ├─► insertActiveNeurons()
│   │   ├─► reinforceConnections() - Hebbian learning for connections
│   │   └─► activateConnections()
│   │
│   └─► activatePatternNeurons() - Hierarchical recognition
│       └─► For each level:
│           ├─► getObservedPatterns() - Detect peaks
│           ├─► matchObservedPatterns() - Match to known patterns
│           ├─► mergeMatchedPatterns() - Hebbian learning for patterns
│           ├─► createNewPatterns() - Create new ◄── REMOVED IN NEW ARCH
│           └─► activateNeurons() at level+1
│
├─► inferNeurons()
│   ├─► reportPredictionsAccuracy()
│   ├─► inferConnections() - All levels at once
│   │   ├─► negativeReinforceConnections()
│   │   └─► Predict from active connections
│   ├─► inferPatterns() - Recursive cascade down
│   ├─► mergeHigherLevelPredictions() - level > 0
│   └─► resolveInputPredictionConflicts() - level 0
│
└─► runForgetCycle()
    └─► Periodic cleanup
```

### New Architecture Flow

```
processFrame(frame, globalReward)
│
├─► applyRewards(globalReward)
│   └─► Strengthen/weaken connections based on reward
│
├─► ageNeurons()
│   └─► Age all context by 1 frame
│
├─► executeOutputs()
│   └─► Execute age=1 predictions
│
├─► recognizeNeurons(frame)
│   ├─► getFrameNeurons() - Find/create base neurons
│   ├─► activateNeurons() - Activate at level 0
│   │   ├─► insertActiveNeurons()
│   │   ├─► reinforceConnections() - Hebbian learning (NO CHANGE)
│   │   └─► activateConnections()
│   │
│   └─► activatePatternNeurons() - Hierarchical recognition
│       └─► For each level:
│           ├─► getObservedPatterns() - Detect peaks
│           ├─► matchObservedPatterns() - Match to known patterns (pattern_past)
│           ├─► mergeMatchedPatterns() - Hebbian learning for patterns (NO CHANGE)
│           └─► activateNeurons() at level+1 ◄── ONLY matching/reinforcement, no creation
│
├─► validateAndLearnFromErrors() ◄── NEW PHASE
│   ├─► Check failed predictions (age=1 in connection_inferred_neurons)
│   ├─► Filter by strength >= minErrorPatternThreshold
│   ├─► For each confident failure:
│   │   ├─► Identify predictor neurons (who made the wrong prediction)
│   │   ├─► For each predictor neuron:
│   │   │   ├─► Create pattern neuron at level+1
│   │   │   ├─► Pattern peak = predictor neuron (not the failed prediction)
│   │   │   ├─► Insert into pattern_past (connections TO predictor)
│   │   │   ├─► Insert into pattern_future (connections FROM predictor)
│   │   │   └─► Create pattern_peaks mapping (pattern → predictor)
│   └─► negativeReinforceConnections() - Weaken failed predictions
│
├─► inferNeurons() ◄── REFACTORED: Top-down flow
│   ├─► reportPredictionsAccuracy()
│   ├─► getMaxActiveLevel()
│   └─► For level = maxLevel down to 0:
│       ├─► inferConnectionsAtLevel(level) - Same-level predictions
│       │   └─► Write to connection_inferred_neurons
│       │
│       └─► If level > 0:
│           ├─► inferPatternsFromLevel(level) - Predict level-1
│           │   └─► Get pattern_future connections
│           │
│           └─► If pattern predictions exist:
│               └─► unpackPatternPredictions(predictions, level-1)
│                   ├─► Get future connections from predicted neurons
│                   ├─► Calculate weighted strengths
│                   ├─► Filter above average
│                   ├─► Write to pattern_inferred_neurons
│                   └─► Recurse down if level-1 > 0
│
└─► runForgetCycle()
    ├─► Decay pattern_past strengths
    ├─► Decay pattern_future strengths
    └─► Periodic cleanup
```

## Pattern Learning: Current vs New

### Current: Continuous Pattern Creation

```
Frame N:
  Active neurons at level 0: [A, B, C]
  │
  ├─► Detect peaks: [C is peak]
  ├─► Match patterns: [No match found]
  └─► Create new pattern: Pattern_1 for peak C
      └─► patterns table: Pattern_1 → {conn(A→C), conn(B→C)}

Frame N+1:
  Active neurons at level 0: [A, B, C]
  │
  ├─► Detect peaks: [C is peak]
  ├─► Match patterns: [Pattern_1 matches 66%]
  └─► Merge (reinforce): Pattern_1 connections strengthened
      └─► patterns table: Pattern_1 → {conn(A→C): 2.0, conn(B→C): 2.0}
```

**Problem:** Creates patterns for every peak, even if not predictive

### New: Error-Driven Pattern Creation

```
Frame N:
  Active neurons at level 0: [A, B, C]
  │
  ├─► Detect peaks: [C is peak]
  ├─► Match patterns: [No match found]
  └─► NO PATTERN CREATION ◄── Key difference
      └─► Only activate matched patterns

Frame N+1:
  Neuron C (active at age=0) predicts Neuron D (strength: 0.8)
  Actual observation: Neuron E (not D)
  │
  └─► validateAndLearnFromErrors():
      ├─► Prediction failed (D not observed)
      ├─► Strength 0.8 >= minErrorPatternThreshold (0.5)
      ├─► Predictor neuron: C (the one that made the wrong prediction)
      └─► Create error pattern at level 1:
          ├─► Pattern_1 neuron created
          ├─► pattern_peaks: Pattern_1 → peak_neuron_id = C (the predictor)
          ├─► pattern_past: Pattern_1 → {connections TO C} (context when C was active)
          └─► pattern_future: Pattern_1 → {connections FROM C} (including C→D that failed)

Frame N+2:
  Neuron C appears again in similar context
  │
  └─► Pattern_1 activated at level 1 (matched via pattern_past)
      └─► Provides top-down prediction via pattern_future
      └─► Pattern learns: "When C appears in this context, here's what it predicts"
```

**Benefit:** Only creates patterns when predictions fail, focusing learning on errors
**Key Insight:** Pattern peak is the predictor neuron (who made the error), not the predicted neuron

## Inference Flow: Current vs New

### Current: Bottom-Up Cascade

```
Level 0 (Base):
  Active neurons: [A, B, C]
  │
  └─► inferConnections() - All levels at once
      ├─► Level 0: Predict [D, E]
      ├─► Level 1: Predict [P1, P2]
      └─► Level 2: Predict [P3]

Then:
  └─► inferPatterns() - Recursive cascade
      ├─► P3 (level 2) → predicts peaks at level 1
      ├─► P2 (level 1) → predicts peaks at level 0
      └─► Cascade down to base level

Result: Base level predictions from both connections and patterns
```

### New: Top-Down Unpacking

```
Level 2 (Highest active):
  Active pattern neurons: [P3]
  │
  ├─► inferConnectionsAtLevel(2) - Same-level predictions
  │   └─► Predict level 2 neurons for next frame
  │
  └─► inferPatternsFromLevel(2) - Predict level 1
      ├─► Get pattern_future connections from P3
      ├─► Predict neurons at level 1: [N1, N2, N3]
      ├─► Filter above average: [N1, N2]
      └─► unpackPatternPredictions([N1, N2], level=1)
          │
          ├─► Get pattern_future connections from N1, N2
          ├─► Predict neurons at level 0: [A, B, C, D]
          ├─► Filter above average: [A, B]
          └─► Write to pattern_inferred_neurons at level 0

Level 1:
  Active neurons: [N1, N2]
  │
  └─► inferConnectionsAtLevel(1) - Same-level predictions
      └─► Predict level 1 neurons for next frame

Level 0 (Base):
  Active neurons: [A, B, C]
  │
  └─► inferConnectionsAtLevel(0) - Same-level predictions
      └─► Predict level 0 neurons for next frame

Final:
  └─► resolveInputPredictionConflicts()
      └─► Merge connection and pattern predictions at level 0
```

**Benefit:** Higher-level context influences lower-level predictions (top-down attention)

## Pattern Table Split: pattern_past vs pattern_future

### Pattern Structure

```
Scenario: Neuron C made a wrong prediction
  Observed sequence: A → B → C (predicted D, but D didn't appear)

Pattern Neuron P1 created at Level 1:
  Peak = C (the predictor neuron that made the error)
  │
  ├─► pattern_past (for recognition):
  │   └─► Connections leading TO peak C (the context when C was active):
  │       ├─► conn(A → C, distance=2)
  │       └─► conn(B → C, distance=1)
  │
  └─► pattern_future (for inference):
      └─► Connections FROM peak C (what C predicts):
          ├─► conn(C → D, distance=1) - the failed prediction
          └─► conn(C → E, distance=2) - other predictions from C
```

### Usage

**Recognition (pattern_past):**
```
Current frame: A and B are active
│
└─► matchObservedPatterns():
    ├─► Detect peak C
    ├─► Get observed connections: {conn(A→C), conn(B→C)}
    ├─► Match against pattern_past:
    │   └─► Pattern P1 has {conn(A→C), conn(B→C)}
    │   └─► Match! Activate P1 at level 1
    └─► NO reinforcement (just activation)
```

**Inference (pattern_future):**
```
P1 is active at level 1
│
└─► inferPatternsFromLevel(1):
    ├─► Get pattern_future connections from P1:
    │   └─► {conn(C→D), conn(C→E)}
    ├─► Predict neurons: [D, E]
    └─► unpackPatternPredictions([D, E], level=0):
        ├─► If D or E are pattern neurons:
        │   └─► Get their pattern_future connections
        │   └─► Continue unpacking recursively
        └─► Write final predictions to pattern_inferred_neurons
```

## Error-Driven Learning Example

### Scenario: Stock Price Prediction

```
Frame 1-5: Learning phase
  Observe: Price UP → Volume HIGH → Price UP
  │
  └─► Connections created (Hebbian):
      ├─► conn(Price_UP → Volume_HIGH, distance=1)
      └─► conn(Volume_HIGH → Price_UP, distance=1)

Frame 6: Prediction
  Observe: Price_UP (active at age=0)
  Predict: Volume_HIGH (strength: 0.9) via connection from Price_UP
  │
  └─► inferConnectionsAtLevel(0):
      └─► connection_inferred_neurons: Volume_HIGH (strength: 0.9)

Frame 7: Validation
  Actual: Volume_LOW observed (not Volume_HIGH)
  │
  └─► validateAndLearnFromErrors():
      ├─► Prediction failed: Volume_HIGH not observed
      ├─► Strength 0.9 >= minErrorPatternThreshold (0.5)
      ├─► Predictor neuron: Price_UP (the one that made the wrong prediction)
      └─► Create error pattern at level 1:
          ├─► Pattern_Market_Context created
          ├─► pattern_peaks: Pattern_Market_Context → Price_UP (the predictor)
          ├─► pattern_past: {connections TO Price_UP} (context when Price_UP was active)
          └─► pattern_future: {connections FROM Price_UP} (including Price_UP → Volume_HIGH)

Frame 8+: Pattern usage
  Similar context occurs, Price_UP appears again
  │
  ├─► Recognition:
  │   └─► Pattern_Market_Context activated (Price_UP is the peak, context matches)
  │   └─► Pattern neuron active at level 1
  │
  └─► Inference:
      └─► Pattern_Market_Context (level 1) uses pattern_future to predict
          └─► Predicts based on connections FROM Price_UP
          └─► Pattern strength reflects reliability of Price_UP's predictions in this context
```

**Key Insight:** Pattern created only when confident prediction failed, capturing the predictor's context
**Pattern Peak:** The neuron that made the error (Price_UP), not the failed prediction (Volume_HIGH)

## Summary of Architectural Benefits

1. **Faster Recognition:** No pattern creation overhead during recognition
2. **Focused Learning:** Patterns created only from prediction errors
3. **Top-Down Inference:** Higher-level context guides lower-level predictions
4. **Biological Plausibility:** Error-driven learning matches cortical learning theories
5. **Reduced Noise:** Fewer spurious patterns from random co-occurrences
6. **Hierarchical Prediction:** Multi-level patterns provide abstract predictions

