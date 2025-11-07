# Error-Driven Learning - Algorithm Analysis

## Test Case: Can It Learn Repeating Sequences?

The ultimate test: A repeating sequence of arbitrary length. The algorithm should initially be inaccurate, but as errors are fixed with patterns, it should approach 100% accuracy.

## Simple Repeating Sequence

### Sequence: `A → B → C → D → A → B → C → D → ...`

#### Initial Frames (Connection Learning)

```
Frame 1: A observed
Frame 2: B observed
  - Connection A→B created (Hebbian)
  
Frame 3: C observed
  - Connection B→C created
  
Frame 4: D observed
  - Connection C→D created
  
Frame 5: A observed
  - Connection D→A created
  - Connection A→B reinforced (Hebbian)
```

After one cycle, we have connections: A→B, B→C, C→D, D→A

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

#### Learning Phase

```
Cycle 1: A → B → C → D
  - Connections created: A→B, B→C, C→D, D→A
  
Cycle 2: A → B → E → F
  - Connections created: B→E, E→F, F→A
  - When B is observed:
    - Predict C (from B→C connection, strength: 1.0)
    - Actual: E appears
    - ERROR! Prediction failed
    - Strength 1.0 >= minErrorPatternThreshold (0.5)
    - Predictor: B (the neuron that predicted C)
    - Create Pattern_1 at level 1:
      - Peak: B
      - pattern_past: {A→B} (connections TO B)
      - pattern_future: {B→C, B→E} (connections FROM B)
```

#### Next Cycle

```
Cycle 3: A → B → C → D
  - A observed, then B observed
  - Pattern_1 activates (B is peak, A→B matches pattern_past)
  - Pattern_1 at level 1 is active
  - Inference:
    - Connection inference: B predicts C and E (both connections exist)
    - Pattern inference: Pattern_1 predicts C and E via pattern_future
  - Actual: C appears
  - Hebbian reinforcement:
    - Connection B→C strengthened
    - Pattern connection to B→C strengthened
  - Negative reinforcement:
    - Connection B→E weakened
    - Pattern connection to B→E weakened
```

#### Next Cycle

```
Cycle 4: A → B → E → F
  - A observed, then B observed
  - Pattern_1 activates (B is peak, A→B matches pattern_past)
  - Inference:
    - Connection B→C (strength: higher after reinforcement)
    - Connection B→E (strength: lower after negative reinforcement)
    - Predict C (stronger)
  - Actual: E appears
  - ERROR AGAIN! Still predicting C when E should appear
```

## The Problem

**The pattern_past is identical in both cases!**

- Case 1: `... → A → B → C` (pattern_past: {A→B})
- Case 2: `... → A → B → E` (pattern_past: {A→B})

Both times B appears after A, so the pattern_past captures the same context. The pattern cannot differentiate between the two cases because the **immediate context is identical**.

### What's Needed

To differentiate, the system needs **longer context**:
- Case 1: `D → A → B → C` (need to know D came before A)
- Case 2: `F → A → B → E` (need to know F came before A)

## Can Higher-Level Patterns Help?

### Hypothesis

Higher-level patterns should capture longer sequences:
- Level 1 pattern captures: `D → A → B` context → predicts C
- Level 1 pattern captures: `F → A → B` context → predicts E

### How Would This Work?

```
After multiple errors at level 0:
  - B keeps predicting wrong (sometimes C, sometimes E)
  - Errors occur with high confidence
  - Patterns created at level 1 with peak = B
  - But all patterns have same pattern_past: {A→B}
  - Patterns can't differentiate!

Question: How do we get patterns with longer context?
```

### Pattern Creation from Level 0 Errors

When B (at level 0) makes an error:
- Create pattern at level 1
- Peak = B
- pattern_past = connections TO B at level 0
- This only captures immediate predecessors (A→B)

**The pattern_past doesn't capture what came before A!**

### Could Level 1 Patterns Help?

If we had level 1 patterns representing longer sequences:
- Pattern_D_A (level 1) represents "D followed by A"
- Pattern_F_A (level 1) represents "F followed by A"

Then when creating error patterns:
- If Pattern_D_A is active when B makes error → different context
- If Pattern_F_A is active when B makes error → different context

But how do these level 1 patterns get created in the first place?

**They're created from level 0 errors!**

This seems circular.

## Potential Solution: Cross-Level Context

When creating an error pattern at level N+1:
- pattern_past should include:
  - Connections TO the predictor at level N (current behavior)
  - **Active patterns at level N+1** (higher-level context)

Example:
```
When B (level 0) makes error:
  - Check if any level 1 patterns are active
  - If Pattern_D_A is active at level 1:
    - Create error pattern with context including Pattern_D_A
  - If Pattern_F_A is active at level 1:
    - Create error pattern with context including Pattern_F_A
```

But this requires level 1 patterns to exist first!

## Bootstrap Problem

1. **Initially:** Only level 0 connections exist
2. **First errors:** Create level 1 patterns from level 0 errors
3. **Problem:** Level 1 patterns have limited context (only immediate connections)
4. **Can't differentiate:** Patterns with same immediate context but different longer context
5. **Need:** Level 2 patterns to provide context to level 1
6. **But:** Level 2 patterns are created from level 1 errors
7. **Circular dependency**

## Alternative: Pattern_Past Should Include Longer History

Instead of pattern_past containing only connections TO the peak, it should contain:
- Connections TO the peak
- Connections TO the neurons that connect TO the peak
- Etc. (longer temporal window)

Example:
```
Sequence: D → A → B → C

Pattern for B predicting C:
  - Peak: B
  - pattern_past: {A→B, D→A} (2-step history)
  - pattern_future: {B→C}

Sequence: F → A → B → E

Pattern for B predicting E:
  - Peak: B
  - pattern_past: {A→B, F→A} (2-step history)
  - pattern_future: {B→E}
```

Now the patterns can differentiate!

But this changes the pattern structure significantly.

## Questions for Review

1. **Is the context problem real?** Can patterns with identical immediate context but different longer context be differentiated?

2. **Should pattern_past capture longer history?** Instead of just connections TO the peak, should it include connections multiple steps back?

3. **Can cross-level patterns solve this?** If level 1 patterns provide context when creating level 2 patterns, does the hierarchy naturally solve the context problem?

4. **Is there a bootstrap issue?** How do higher-level patterns get created if lower-level patterns can't differentiate contexts?

5. **Does the current design work for simple cases?** Even if it can't handle complex context-dependent sequences, will it work for simpler patterns?

## Possible Answers

### Answer 1: Pattern Distance Parameter

Connections have a `distance` parameter. Maybe patterns should too?

When creating a pattern from error:
- Include connections with distance <= some threshold
- This captures longer temporal context

Example:
```
When B makes error:
  - Get connections TO B with distance=1: {A→B}
  - Get connections TO B with distance=2: {D→A→B} or {F→A→B}
  - pattern_past includes both
```

But connections are point-to-point, not chains. We'd need to traverse the connection graph.

### Answer 2: Active Neuron History

When creating a pattern, include not just connections TO the peak, but the **active neurons** at the time:

```
When B makes error:
  - Active neurons (age=0): B
  - Active neurons (age=1): A
  - Active neurons (age=2): D or F
  - pattern_past captures: {B, A, D} or {B, A, F}
```

This gives patterns different contexts based on longer history.

### Answer 3: Trust the Hierarchy

Maybe the algorithm works as designed:
1. Level 0 learns simple connections
2. Level 1 patterns capture immediate context (limited)
3. Level 2 patterns capture level 1 context (longer)
4. Level 3 patterns capture level 2 context (even longer)
5. Hierarchy naturally builds longer context

The question is: **Will this converge to 100% accuracy?**

### Answer 4: Pattern Matching Includes Active Context

When matching patterns during recognition:
- Don't just match pattern_past connections
- Also check what higher-level patterns are active
- Pattern activates only if both connection context AND higher-level context match

This would allow patterns to differentiate based on hierarchical context.

## My Current Assessment

**The algorithm as currently designed may not reach 100% accuracy for context-dependent sequences** because:

1. Patterns created from errors only capture immediate context (connections TO predictor)
2. If two different contexts have the same immediate predecessor, patterns can't differentiate
3. Higher-level patterns are created from lower-level errors, but face the same context limitation
4. No mechanism to capture longer temporal context in pattern_past

**Possible solutions:**
1. Include longer connection history in pattern_past (traverse connection graph)
2. Include active neuron history when creating patterns
3. Include active higher-level patterns as context when creating patterns
4. Trust that the hierarchy will naturally solve this (needs verification)

**Recommendation:** Test with the context-dependent sequence example to see what actually happens.

