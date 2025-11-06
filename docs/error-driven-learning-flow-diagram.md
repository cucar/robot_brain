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
│   │   ├─► reinforceConnections() ◄── REMOVED IN NEW ARCH
│   │   └─► activateConnections()
│   │
│   └─► activatePatternNeurons() - Hierarchical recognition
│       └─► For each level:
│           ├─► getObservedPatterns() - Detect peaks
│           ├─► matchObservedPatterns() - Match to known patterns
│           ├─► mergeMatchedPatterns() - Reinforce ◄── REMOVED IN NEW ARCH
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
│   │   ├─► createConnections() ◄── NEW: Hebbian only, no reinforcement
│   │   └─► activateConnections()
│   │
│   └─► activatePatternNeurons() - Hierarchical recognition
│       └─► For each level:
│           ├─► getObservedPatterns() - Detect peaks
│           ├─► matchObservedPatterns() - Match to known patterns (pattern_past)
│           └─► activateNeurons() at level+1 ◄── ONLY matching, no creation
│
├─► validateAndLearnFromErrors() ◄── NEW PHASE
│   ├─► Check failed predictions (age=1 in connection_inferred_neurons)
│   ├─► Filter by strength >= minErrorPatternThreshold
│   ├─► For each confident failure:
│   │   ├─► Get connections from connection_inference
│   │   ├─► Create pattern neuron at level+1
│   │   ├─► Insert into pattern_past (connections TO peak)
│   │   ├─► Insert into pattern_future (connections FROM peak)
│   │   └─► Create pattern_peaks mapping
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
  Prediction from connections: Neuron D (strength: 0.8)
  Actual observation: Neuron E
  │
  └─► validateAndLearnFromErrors():
      ├─► Prediction failed (D not observed)
      ├─► Strength 0.8 >= minErrorPatternThreshold (0.5)
      └─► Create error pattern at level 1:
          ├─► Pattern_1 neuron created
          ├─► pattern_past: Pattern_1 → {connections that predicted D}
          ├─► pattern_future: Pattern_1 → {connections from D}
          └─► pattern_peaks: Pattern_1 → peak_neuron_id = D

Frame N+2:
  Similar context occurs again
  │
  └─► Pattern_1 activated at level 1
      └─► Provides top-down prediction via pattern_future
```

**Benefit:** Only creates patterns when predictions fail, focusing learning on errors

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
Peak Neuron C at Level 0:
  Observed sequence: A → B → C → D → E
  
Pattern Neuron P1 created at Level 1:
  │
  ├─► pattern_past (for recognition):
  │   └─► Connections leading TO peak C:
  │       ├─► conn(A → C, distance=2)
  │       └─► conn(B → C, distance=1)
  │
  └─► pattern_future (for inference):
      └─► Connections FROM peak C:
          ├─► conn(C → D, distance=1)
          └─► conn(C → E, distance=2)
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
  Observe: Price UP
  Predict: Volume HIGH (strength: 0.9)
  │
  └─► inferConnectionsAtLevel(0):
      └─► connection_inferred_neurons: Volume_HIGH (strength: 0.9)

Frame 7: Validation
  Actual: Volume LOW (not Volume HIGH)
  │
  └─► validateAndLearnFromErrors():
      ├─► Prediction failed: Volume_HIGH not observed
      ├─► Strength 0.9 >= minErrorPatternThreshold (0.5)
      └─► Create error pattern at level 1:
          ├─► Pattern_Market_Reversal created
          ├─► pattern_past: {conn(Price_UP → Volume_HIGH)}
          ├─► pattern_future: {conn(Volume_HIGH → Price_UP)}
          └─► pattern_peaks: Pattern_Market_Reversal → Volume_HIGH

Frame 8+: Pattern usage
  Similar context: Price UP observed
  │
  ├─► Recognition:
  │   └─► Pattern_Market_Reversal NOT activated (Volume_HIGH not observed)
  │
  └─► Inference:
      └─► If Pattern_Market_Reversal was active at level 1:
          └─► Would predict Price_UP via pattern_future
```

**Key Insight:** Pattern created only when confident prediction failed, capturing the error context for future learning

## Summary of Architectural Benefits

1. **Faster Recognition:** No pattern creation overhead during recognition
2. **Focused Learning:** Patterns created only from prediction errors
3. **Top-Down Inference:** Higher-level context guides lower-level predictions
4. **Biological Plausibility:** Error-driven learning matches cortical learning theories
5. **Reduced Noise:** Fewer spurious patterns from random co-occurrences
6. **Hierarchical Prediction:** Multi-level patterns provide abstract predictions

