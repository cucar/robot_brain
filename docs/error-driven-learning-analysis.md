# Error-Driven Learning - Algorithm Analysis

## Executive Summary

**Question:** Will the error-driven learning algorithm memorize repeating sequences and approach 100% accuracy?

**Answer:** YES! ✅

**Key Reasons:**
1. **Multiple-distance connections** capture long temporal context (up to 9 frames back)
2. **Pattern_past includes ALL connections** to the peak neuron at all distances
3. **Different contexts have different active neurons** at various distances
4. **Multiple patterns per neuron** handle different contexts independently
5. **Hebbian learning** refines pattern predictions over time
6. **Hierarchical patterns** extend context to arbitrary lengths

**Limitation:** Truly random/unpredictable sequences won't be memorized (correct behavior - can't learn randomness)

**Conclusion:** The algorithm will converge to 100% accuracy for any sequence with learnable temporal structure.

---

## Test Case: Can It Learn Repeating Sequences?

The ultimate test: A repeating sequence of arbitrary length. The algorithm should initially be inaccurate, but as errors are fixed with patterns, it should approach 100% accuracy.

## Key Understanding: Multiple Distance Connections

**CRITICAL:** The brain learns connections at ALL distances, not just distance=1!

From `reinforceConnections()`:
```javascript
SELECT
    f.neuron_id as from_neuron_id,
    t.neuron_id as to_neuron_id,
    f.age as distance,  // <-- Distance = age of source neuron
    1 as strength
FROM active_neurons f
CROSS JOIN active_neurons t
WHERE t.age = 0  // target is newly activated
```

This means:
- Neuron at age=0 connects to new neuron (distance=0, self-connection context)
- Neuron at age=1 connects to new neuron (distance=1)
- Neuron at age=2 connects to new neuron (distance=2)
- ... up to age=9 (distance=9)

**Every active neuron creates a connection to every newly activated neuron!**

## Simple Repeating Sequence

### Sequence: `A → B → C → D → A → B → C → D → ...`

#### Initial Frames (Connection Learning)

```
Frame 1: A observed (age=0)

Frame 2: B observed (age=0)
  Active neurons: A (age=1), B (age=0)
  Connections created:
    - A→B (distance=1)

Frame 3: C observed (age=0)
  Active neurons: A (age=2), B (age=1), C (age=0)
  Connections created:
    - A→C (distance=2)
    - B→C (distance=1)

Frame 4: D observed (age=0)
  Active neurons: A (age=3), B (age=2), C (age=1), D (age=0)
  Connections created:
    - A→D (distance=3)
    - B→D (distance=2)
    - C→D (distance=1)

Frame 5: A observed (age=0)
  Active neurons: B (age=3), C (age=2), D (age=1), A (age=0)
  Connections created:
    - B→A (distance=3)
    - C→A (distance=2)
    - D→A (distance=1)
  Connections reinforced:
    - A→B, A→C, A→D (from previous cycle)
```

After one cycle, we have connections at MULTIPLE distances!

#### Prediction Phase

```
Frame 6: B observed
  - Predict C (from B→C connection) ✓ CORRECT
  
Frame 7: C observed
  - Predict D (from C→D connection) ✓ CORRECT
  
Frame 8: D observed
  - Predict A (from D→A connection) ✓ CORRECT
  
Frame 9: A observed
  - Predict B (from A→B connection) ✓ CORRECT
```

### Observation

**The sequence will be learned perfectly by connections alone!**

If the sequence is deterministic and repeating, connection inference will achieve 100% accuracy without ever creating patterns, because:
- No prediction errors occur
- No patterns are created
- System stays at level 0

**This is fine!** Simple sequences don't need patterns.

## Context-Dependent Sequence (The Real Test)

### Sequence: `A → B → C → D → A → B → E → F → A → B → C → D → A → B → E → F → ...`

This sequence has ambiguity: After `A → B`, sometimes `C` appears, sometimes `E` appears.

**Key question:** Can the brain differentiate based on what came BEFORE the `A → B`?
- First case: `D → A → B → C` (D precedes the A→B→C sequence)
- Second case: `F → A → B → E` (F precedes the A→B→E sequence)

#### Learning Phase - WITH MULTIPLE DISTANCES

```
Cycle 1: A → B → C → D
  Frame 1: A (age=0)
  Frame 2: B (age=0), A (age=1)
    Connections: A→B (distance=1)
  Frame 3: C (age=0), B (age=1), A (age=2)
    Connections: A→C (distance=2), B→C (distance=1)
  Frame 4: D (age=0), C (age=1), B (age=2), A (age=3)
    Connections: A→D (distance=3), B→D (distance=2), C→D (distance=1)

Cycle 2: A → B → E → F
  Frame 5: A (age=0), D (age=1), C (age=2), B (age=3)
    Connections: D→A (distance=1), C→A (distance=2), B→A (distance=3)
  Frame 6: B (age=0), A (age=1), D (age=2), C (age=3)
    Connections: A→B (distance=1) [reinforced], D→B (distance=2), C→B (distance=3)
  Frame 7: E (age=0), B (age=1), A (age=2), D (age=3)
    Connections: B→E (distance=1), A→E (distance=2), D→E (distance=3)

    At this frame, B predicted C (from B→C connection)
    But E appeared instead!
    ERROR! Create pattern from this error.
```

#### Error Pattern Creation - WITH MULTIPLE DISTANCES

```
When B predicted C but E appeared:
  - Predictor: B (the neuron that made wrong prediction)
  - Create Pattern_1 at level 1:
    - Peak: B
    - pattern_past: ALL connections TO B that exist
      - A→B (distance=1)
      - D→B (distance=2)  ← KEY! D is in the context!
      - C→B (distance=3)
    - pattern_future: ALL connections FROM B
      - B→C (distance=1)
      - B→D (distance=2)
      - B→E (distance=1)
      - B→A (distance=3)
```

**CRITICAL INSIGHT:** The pattern_past includes D→B (distance=2)!

This means the pattern captures that D was active 2 frames before B appeared.

#### Next Cycle - First Case

```
Cycle 3: A → B → C → D
  Frame N: D (age=0)
  Frame N+1: A (age=0), D (age=1)
  Frame N+2: B (age=0), A (age=1), D (age=2)

    Active neurons: B (age=0), A (age=1), D (age=2)

    Pattern_1 matching:
      - Peak: B ✓ (B is active at age=0)
      - pattern_past connections:
        - A→B (distance=1) ✓ (A is at age=1)
        - D→B (distance=2) ✓ (D is at age=2)
        - C→B (distance=3) ✗ (C is NOT active)

    Pattern matches! (enough connections match)
    Pattern_1 activates at level 1
```

#### Next Cycle - Second Case

```
Cycle 4: A → B → E → F
  Frame M: F (age=0)
  Frame M+1: A (age=0), F (age=1)
  Frame M+2: B (age=0), A (age=1), F (age=2)

    Active neurons: B (age=0), A (age=1), F (age=2)

    Pattern_1 matching:
      - Peak: B ✓ (B is active at age=0)
      - pattern_past connections:
        - A→B (distance=1) ✓ (A is at age=1)
        - D→B (distance=2) ✗ (D is NOT active, F is at age=2)
        - C→B (distance=3) ✗ (C is NOT active)

    Pattern does NOT match well (D is missing)
    Pattern_1 does NOT activate (or activates weakly)
```

## The Solution: Multiple Distance Connections!

**The pattern_past DOES capture longer context** because connections exist at multiple distances!

When B makes an error:
- pattern_past includes A→B (distance=1)
- pattern_past includes D→B (distance=2)
- pattern_past includes C→B (distance=3)

This captures what happened 1, 2, and 3 frames before B appeared!

**Different contexts will have different active neurons at different distances:**
- Context 1: D at distance=2 when B appears
- Context 2: F at distance=2 when B appears

The patterns can differentiate!

## How Patterns Evolve Over Time

### Initial Error (Cycle 2)

```
Pattern_1 created with:
  - pattern_past: {A→B, D→B, C→B}
  - pattern_future: {B→C, B→D, B→E, B→A}
  - All connections have initial strength
```

### After Cycle 3 (D → A → B → C)

```
Pattern_1 activated (D was at distance=2)
  - Actual: C appeared
  - Hebbian reinforcement:
    - Pattern connection to B→C strengthened
  - Negative reinforcement:
    - Pattern connection to B→E weakened
    - Pattern connection to B→D weakened
    - Pattern connection to B→A weakened
```

### After Cycle 4 (F → A → B → E)

```
Pattern_1 does NOT activate (F at distance=2, not D)
  - Connection B→E gets reinforced (Hebbian at connection level)
  - But Pattern_1 doesn't learn from this
  - Need a DIFFERENT pattern for this context!
```

### New Error Pattern Created

```
When B predicts C but E appears (in F→A→B→E context):
  - Create Pattern_2:
    - Peak: B
    - pattern_past: {A→B, F→B, E→B, ...}
    - pattern_future: {B→C, B→E, ...}
```

Now we have TWO patterns for B:
- Pattern_1: Activates when D is at distance=2 → predicts C
- Pattern_2: Activates when F is at distance=2 → predicts E

### Convergence

Over many cycles:
- Pattern_1 strengthens B→C, weakens others
- Pattern_2 strengthens B→E, weakens others
- Each pattern learns its specific context
- Predictions become accurate!

## The Truly Unpredictable Case

### Sequence: `A → B → C` (50% of the time) and `A → B → E` (50% of the time)

**With NO correlation to previous sequences** - completely random which one occurs.

```
Frame N-2: X (random)
Frame N-1: A
Frame N: B
Frame N+1: C or E (50/50 random)
```

**Can this be learned?** NO! And that's fine!

There's no pattern to learn. The brain will:
1. Create connections B→C and B→E
2. Both will have similar strength (reinforced equally often)
3. Predictions will be uncertain (both predicted with ~50% strength)
4. Errors will occur, patterns created
5. But patterns can't improve accuracy (no differentiating context)
6. System settles at ~50% accuracy for this ambiguous case

**This is correct behavior!** The brain shouldn't memorize randomness.

### But in Real Data...

The probability of hitting a case where:
- Preceding conditions are EXACTLY the same
- But outcomes are different
- With NO correlation to any earlier context

...is **infinitesimally small** in real-world data!

Real sequences have structure:
- Different preceding patterns (captured by multiple-distance connections)
- Different higher-level contexts (captured by pattern hierarchy)
- Temporal correlations at various time scales

## Will The Brain Memorize Learnable Sequences?

**YES!** Here's why:

### 1. Multiple Distance Connections Capture Long Context

Connections exist at distances 1-9 (up to baseNeuronMaxAge):
- Distance=1: Immediate predecessor
- Distance=2: 2 frames back
- Distance=3: 3 frames back
- ... up to distance=9

When creating error patterns, pattern_past includes ALL these connections, capturing up to 9 frames of history!

### 2. Pattern Matching Uses This Context

When matching patterns during recognition:
- Check if peak neuron is active
- Check if pattern_past connections match active neurons at correct distances
- Pattern only activates if context matches

Different contexts → different active neurons at various distances → different pattern activation

### 3. Multiple Patterns Per Neuron

A single neuron can be the peak of MULTIPLE patterns:
- Pattern_1: B in context {D at distance=2} → predicts C
- Pattern_2: B in context {F at distance=2} → predicts E
- Pattern_3: B in context {X at distance=3} → predicts Y

Each pattern learns independently!

### 4. Hebbian Learning Refines Patterns

Over time:
- Correct predictions strengthen pattern connections
- Incorrect predictions weaken pattern connections
- Patterns converge to accurate predictions for their specific contexts

### 5. Hierarchical Patterns Extend Context

If 9 frames isn't enough:
- Level 1 patterns represent sequences at level 0
- Level 2 patterns represent sequences of level 1 patterns
- Each level extends the temporal context exponentially
- Level 2 can represent sequences spanning many more frames

## Final Assessment

**The algorithm WILL memorize sequences that can be predicted based on temporal patterns!**

Key insights:
1. ✅ Multiple-distance connections capture long temporal context (up to 9 frames)
2. ✅ Pattern_past includes all connections TO peak, capturing full context
3. ✅ Different contexts activate different patterns (or same pattern differently)
4. ✅ Multiple patterns per neuron handle different contexts
5. ✅ Hebbian learning refines pattern predictions over time
6. ✅ Hierarchy extends context to arbitrary lengths
7. ✅ Truly random/unpredictable sequences won't be memorized (correct behavior)

**The algorithm should converge to 100% accuracy for any sequence that has learnable temporal structure!**

