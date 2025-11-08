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
  Active neurons at level 0: [A (age=2), B (age=1), C (age=0)]
  │
  ├─► Detect peaks: [C is peak]
  ├─► Match patterns: [No match found]
  └─► Create new pattern: Pattern_1 for peak C
      └─► patterns table: Pattern_1 → {conn(A→C, dist=2), conn(B→C, dist=1)}

Frame N+1:
  Active neurons at level 0: [A (age=2), B (age=1), C (age=0)]
  │
  ├─► Detect peaks: [C is peak]
  ├─► Match patterns: [Pattern_1 matches 66%]
  └─► Merge (reinforce): Pattern_1 connections strengthened
      └─► patterns table: Pattern_1 → {conn(A→C, dist=2): 2.0, conn(B→C, dist=1): 2.0}
```

**Problem:** Creates patterns for every peak, even if not predictive

**Note:** Connections exist at multiple distances (1-9 based on source neuron age), capturing temporal context

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
          ├─► pattern_past: Pattern_1 → {ALL connections TO C at distances 1-9}
          │   Example: {A→C (dist=2), B→C (dist=1), X→C (dist=3), ...}
          └─► pattern_future: Pattern_1 → {ALL connections FROM C}
              Example: {C→D (dist=1), C→E (dist=2), ...}

Frame N+2:
  Neuron C appears again in similar context
  Active neurons: [A (age=2), B (age=1), C (age=0)]
  │
  └─► Pattern_1 matching:
      ├─► Peak C is active ✓
      ├─► Check pattern_past: A at age=2 ✓, B at age=1 ✓
      └─► Pattern_1 activated at level 1 (context matches!)
          └─► Provides top-down prediction via pattern_future
          └─► Pattern learns: "When C appears in this context, here's what it predicts"
```

**Benefit:** Only creates patterns when predictions fail, focusing learning on errors
**Key Insight:** Pattern peak is the predictor neuron (who made the error), not the predicted neuron
**Context Capture:** pattern_past includes connections at ALL distances (1-9), capturing up to 9 frames of history

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
  Observed sequence: X → Y → A → B → C (predicted D, but D didn't appear)
  Active neurons when C appeared: X (age=4), Y (age=3), A (age=2), B (age=1), C (age=0)

Pattern Neuron P1 created at Level 1:
  Peak = C (the predictor neuron that made the error)
  │
  ├─► pattern_past (for recognition):
  │   └─► ALL connections leading TO peak C (the context when C was active):
  │       ├─► conn(B → C, distance=1) - immediate predecessor
  │       ├─► conn(A → C, distance=2) - 2 frames back
  │       ├─► conn(Y → C, distance=3) - 3 frames back
  │       ├─► conn(X → C, distance=4) - 4 frames back
  │       └─► ... up to distance=9 (captures 9 frames of history!)
  │
  └─► pattern_future (for inference):
      └─► ALL connections FROM peak C (what C predicts):
          ├─► conn(C → D, distance=1) - the failed prediction
          ├─► conn(C → E, distance=2) - other predictions from C
          └─► ... all connections FROM C
```

**Key Insight:** pattern_past captures up to 9 frames of temporal context through multiple-distance connections!

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

### Scenario: Stock Price Prediction with Context

```
Frames 1-10: Learning phase
  Observe: News_Positive → Price_UP → Volume_HIGH → Price_UP
  │
  └─► Connections created (Hebbian) at multiple distances:
      ├─► conn(News_Positive → Price_UP, distance=1)
      ├─► conn(News_Positive → Volume_HIGH, distance=2)
      ├─► conn(Price_UP → Volume_HIGH, distance=1)
      ├─► conn(Volume_HIGH → Price_UP, distance=1)
      └─► ... many more at various distances

Frame 11: Different context
  Observe: News_Negative → Price_UP (dead cat bounce)
  Active neurons: News_Negative (age=1), Price_UP (age=0)
  Predict: Volume_HIGH (strength: 0.9) via connection from Price_UP
  │
  └─► inferConnectionsAtLevel(0):
      └─► connection_inferred_neurons: Volume_HIGH (strength: 0.9)

Frame 12: Validation
  Actual: Volume_LOW observed (not Volume_HIGH)
  │
  └─► validateAndLearnFromErrors():
      ├─► Prediction failed: Volume_HIGH not observed
      ├─► Strength 0.9 >= minErrorPatternThreshold (0.5)
      ├─► Predictor neuron: Price_UP (the one that made the wrong prediction)
      └─► Create error pattern at level 1:
          ├─► Pattern_Bearish_Context created
          ├─► pattern_peaks: Pattern_Bearish_Context → Price_UP (the predictor)
          ├─► pattern_past: {ALL connections TO Price_UP}
          │   ├─► News_Negative → Price_UP (distance=1)
          │   └─► ... other active neurons at various distances
          └─► pattern_future: {ALL connections FROM Price_UP}
              ├─► Price_UP → Volume_HIGH (distance=1) - the failed prediction
              └─► ... other predictions from Price_UP

Frame 13+: Pattern usage - Bullish context
  Observe: News_Positive → Price_UP
  Active neurons: News_Positive (age=1), Price_UP (age=0)
  │
  ├─► Recognition:
  │   └─► Pattern_Bearish_Context does NOT activate
  │       (News_Positive at distance=1, not News_Negative)
  │
  └─► Inference:
      └─► Connection-based prediction: Volume_HIGH (works in bullish context)

Frame 20+: Pattern usage - Bearish context
  Observe: News_Negative → Price_UP
  Active neurons: News_Negative (age=1), Price_UP (age=0)
  │
  ├─► Recognition:
  │   └─► Pattern_Bearish_Context ACTIVATES ✓
  │       (News_Negative at distance=1 matches pattern_past)
  │   └─► Pattern neuron active at level 1
  │
  └─► Inference:
      └─► Pattern_Bearish_Context provides top-down prediction
          └─► Predicts Volume_LOW (learned from error)
          └─► Overrides or modulates connection-based prediction
```

**Key Insight:** Pattern differentiates contexts based on what happened BEFORE Price_UP appeared
**Context Differentiation:** News_Positive vs News_Negative at distance=1 creates different contexts
**Pattern Peak:** The neuron that made the error (Price_UP), not the failed prediction (Volume_HIGH)
**Multiple Distance Power:** Captures up to 9 frames of history, enabling rich context differentiation

## Summary of Architectural Benefits

1. **Faster Recognition:** No pattern creation overhead during recognition
2. **Focused Learning:** Patterns created only from prediction errors
3. **Top-Down Inference:** Higher-level context guides lower-level predictions
4. **Biological Plausibility:** Error-driven learning matches cortical learning theories
5. **Reduced Noise:** Fewer spurious patterns from random co-occurrences
6. **Hierarchical Prediction:** Multi-level patterns provide abstract predictions

