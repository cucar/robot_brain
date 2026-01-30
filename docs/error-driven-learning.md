# Error-Driven Learning

## Overview

Error-driven learning is the mechanism by which the brain creates patterns. Patterns are **only** created when confident predictions fail - not during normal recognition. This produces sparse, meaningful patterns focused on correcting prediction errors rather than memorizing noise.

**Key Properties:**
- Patterns created only when prediction strength >= threshold AND prediction fails
- The **predictor neuron** (not the failed prediction) becomes the pattern's peak
- Pattern captures the temporal context when the error occurred
- Pattern stores the correct prediction for future use
- Multiple patterns per neuron handle different contexts independently

---

## When Patterns Are Created

Patterns are created by `learnNewPatterns()` in two scenarios:

### 1. Event Prediction Errors

When a connection confidently predicted an event that didn't happen:
- Connection strength >= `eventErrorMinStrength` (default: 2.0)
- The predicted neuron did NOT appear
- Create a pattern with the predictor as peak

### 2. Action Regret

When an action resulted in negative reward:
- Pattern strength >= `actionRegretMinStrength` (default: 2.0)
- The action's reward was negative
- Create a pattern to try alternative actions

---

## Pattern Structure

When neuron C predicts D but E appears instead:

```
Pattern Created at Level 1:

  Peak Neuron: C (the predictor, not D the failed prediction)

  pattern_past (context for recognition):
    - B at context_age=1 (was at age=1 when C appeared)
    - A at context_age=2 (was at age=2 when C appeared)
    - X at context_age=3 (was at age=3 when C appeared)
    - Captures up to contextLength-1 frames of temporal history

  pattern_future (predictions):
    - E at distance=1 (the actual outcome)

  pattern_peaks:
    - Maps pattern_neuron_id → peak_neuron_id (C)
```

**Why the predictor is the peak:** The predictor neuron made the error, so it needs to learn. When C appears again in a similar context, the pattern activates and provides the corrected prediction.

---

## Example: Learning a Context-Dependent Sequence

### Sequence: `A → B → C → D → A → B → E → F → ...` (repeating)

After `A → B`, sometimes `C` appears (when D preceded), sometimes `E` appears (when F preceded).

### Initial Learning (Connections Only)

```
Cycle 1: A → B → C → D
  Connections created: A→B, A→C, B→C, A→D, B→D, C→D

Cycle 2: A → B → E → F
  Frame 7: E appears
    B predicted C (from B→C connection, strength >= threshold)
    But E appeared instead!

    ERROR → Create Pattern_1:
      Peak: B
      pattern_past: {A at age=1, D at age=2, C at age=3}
      pattern_future: {E at distance=1}
```

### Pattern Recognition in Future Cycles

```
Cycle 3: D → A → B → ?
  Frame N+2: B appears with A at age=1, D at age=2

  Pattern_1 matching:
    - Peak B active at age=0 ✓
    - A at age=1 ✓
    - D at age=2 ✓
    - C at age=3 ✗ (missing)

  Match ratio: 2/3 = 67% >= mergePatternThreshold (50%)
  Pattern_1 activates!
  Pattern predicts E (overriding connection's prediction of C)
```

### Context Differentiation

```
Cycle 4: F → A → B → ?
  Frame M+2: B appears with A at age=1, F at age=2

  Pattern_1 matching:
    - Peak B active at age=0 ✓
    - A at age=1 ✓
    - D at age=2 ✗ (F is there instead)
    - C at age=3 ✗

  Match ratio: 1/3 = 33% < mergePatternThreshold (50%)
  Pattern_1 does NOT activate
  Connection inference predicts (may create Pattern_2 on error)
```

Over time, two patterns emerge for B:
- **Pattern_1**: Context includes D → predicts C
- **Pattern_2**: Context includes F → predicts E

---

## How Patterns Are Used

### Recognition Phase

When a peak neuron appears, `matchObservedPatterns()` checks if any patterns match:

```
Active neurons: Y (age=3), A (age=2), B (age=1), C (age=0)

Pattern for peak C:
  pattern_past: {B at context_age=1, A at context_age=2, X at context_age=3}

Matching:
  - B at context_age=1? B is at age=1 ✓
  - A at context_age=2? A is at age=2 ✓
  - X at context_age=3? Y is at age=3 ✗

Match ratio: 2/3 = 67% >= mergePatternThreshold (50%)
Pattern activates!
```

Among matching patterns for the same peak, the one with highest total strength wins.

### Inference Phase

When a pattern is active, `collectVotes()` gathers its predictions:

1. Get pattern_future entries where `distance = pattern.age + 1`
2. Cast votes weighted by level and time decay
3. Pattern votes **override** connection votes from the peak neuron

This override is the key mechanism: patterns exist to correct connection predictions.

---

## Multiple-Distance Connections

Connections are created at ALL distances within the context window:

```
Frame 4: D observed (age=0)
  Active neurons: A (age=3), B (age=2), C (age=1), D (age=0)

  Connections created:
    - A→D (distance=3)
    - B→D (distance=2)
    - C→D (distance=1)
```

This means pattern_past captures the full temporal context - not just the immediate predecessor, but what happened 2, 3, 4... frames ago.

**Simple sequences don't need patterns.** If a sequence is deterministic (A→B→C→D repeating), connections alone achieve 100% accuracy. Patterns are only created when connections make errors.

---

## Context Differentiation

Consider: `A → B → C → D → A → B → E → F → ...` (repeating)

After `A → B`, sometimes C appears (preceded by D), sometimes E appears (preceded by F).

When B predicts C but E appears:
- Pattern created with peak=B
- pattern_past includes D at context_age=2

Later, when B appears with D at age=2:
- Pattern matches, predicts E (overriding connection's prediction of C)

When B appears with F at age=2:
- Pattern doesn't match (D missing)
- A different pattern handles this context

**Different contexts = different active neurons at different ages = different patterns activate.**

---

## Pattern Evolution

Patterns refine over time through `refinePatterns()`:

**pattern_past refinement:**
- Common neurons (in both pattern and observation): strengthen
- Novel neurons (only in observation): add with strength 1
- Missing neurons (only in pattern): weaken

**pattern_future refinement:**
- Correct predictions: strengthen
- Failed predictions: weaken
- Novel observations: add

Over many cycles, patterns converge to accurate predictions for their specific contexts.

---

## Unpredictable Sequences

For truly random sequences (A→B→C 50%, A→B→E 50% with no correlation to history):

- Connections B→C and B→E both form with similar strength
- Patterns are created but can't differentiate (same context)
- System settles at ~50% accuracy

**This is correct behavior.** The brain shouldn't memorize randomness. Real-world data has structure that patterns can exploit.

---

## Multiple Patterns Per Neuron

A single neuron can be the peak of multiple patterns:

```
Pattern_1: peak=B, context includes D at age=2 → predicts C
Pattern_2: peak=B, context includes F at age=2 → predicts E
Pattern_3: peak=B, context includes X at age=3 → predicts Y
```

Each pattern learns independently. When B appears, the pattern with the best-matching context activates.

---

## Hierarchical Patterns

If contextLength frames isn't enough context:

- Level 1 patterns capture sequences of base neurons
- Level 2 patterns capture sequences of Level 1 patterns
- Each level extends the effective temporal context
- Higher levels can represent patterns spanning many more frames

Pattern errors at level N create patterns at level N+1.

---

## Summary

| Concept | Description |
|---------|-------------|
| **Pattern creation** | Only on confident prediction errors |
| **Peak neuron** | The predictor that made the error |
| **pattern_past** | Context neurons with relative ages |
| **pattern_future** | Corrected predictions |
| **Pattern matching** | Threshold-based (default 50%) |
| **Pattern override** | Pattern votes replace connection votes |
| **Context differentiation** | Different histories → different patterns |
| **Hierarchical extension** | Pattern errors create higher-level patterns |

The algorithm memorizes any sequence with learnable temporal structure, while correctly refusing to memorize true randomness.
