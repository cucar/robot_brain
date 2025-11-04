# Error-Based Pattern Creation: Visual Guide

## The Problem: Context-Dependent Predictions

```
Sequence A: A → B → C → D
Sequence B: A → B → C → Z

After seeing "ABC", should we predict D or Z?
Connection inference alone cannot distinguish!
```

---

## The Solution: Error-Based Patterns

### Episode 1: Learn Sequence A

```
Frame 1: A
Frame 2: B    [Learn: A→B]
Frame 3: C    [Learn: A→C, B→C]
Frame 4: D    [Learn: A→D, B→D, C→D]

No errors → No patterns created
```

### Episode 2: See Sequence B (First Error!)

```
Frame 1: A
Frame 2: B    [Predict: B ✓]
Frame 3: C    [Predict: C ✓]
Frame 4: Z    [Predict: D ✗]  ← ERROR!

Error detected:
- False positive: Predicted D, but D didn't activate
- False negative: Didn't predict Z, but Z activated
```

### Pattern Creation from Error

```
Create Pattern P1:
┌─────────────────────────────────────┐
│ Pattern Neuron: P1 (at level 1)    │
│ Peak Neuron: C (at level 0)        │
├─────────────────────────────────────┤
│ PAST CONNECTIONS (context):        │
│   A→B (dist=1)                     │
│   A→C (dist=2)                     │
│   B→C (dist=1)                     │
├─────────────────────────────────────┤
│ FUTURE CONNECTIONS (prediction):   │
│   A→Z (dist=3)                     │
│   B→Z (dist=2)                     │
│   C→Z (dist=1)                     │
└─────────────────────────────────────┘

Meaning: "When you see A→B→C, predict Z"
```

### Episode 3: See Sequence A Again (Pattern Activates!)

```
Frame 1: A
Frame 2: B
Frame 3: C

Check if P1 should activate:
  ✓ A→B active? YES (A at age=2, B at age=1)
  ✓ A→C active? YES (A at age=2, C at age=0)
  ✓ B→C active? YES (B at age=1, C at age=0)
  
→ All past connections active!
→ P1 ACTIVATES at level 1

P1 predicts its future connections:
  - A→Z (strength=1.0)
  - B→Z (strength=1.0)
  - C→Z (strength=1.0)

Frame 4: D    [Predict: D (from base) + Z (from P1)]

Error detected:
- False positive: Predicted Z, but Z didn't activate
- False negative: Didn't predict D strongly enough

Create Pattern P2 for D...
```

### Pattern P2 Creation

```
Create Pattern P2:
┌─────────────────────────────────────┐
│ Pattern Neuron: P2 (at level 1)    │
│ Peak Neuron: C (at level 0)        │
├─────────────────────────────────────┤
│ PAST CONNECTIONS (context):        │
│   A→B (dist=1)  ← SAME AS P1!     │
│   A→C (dist=2)  ← SAME AS P1!     │
│   B→C (dist=1)  ← SAME AS P1!     │
├─────────────────────────────────────┤
│ FUTURE CONNECTIONS (prediction):   │
│   A→D (dist=3)                     │
│   B→D (dist=2)                     │
│   C→D (dist=1)                     │
└─────────────────────────────────────┘

Pattern Matching:
  P2 has same peak (C) as P1
  P2 has same past connections as P1
  → 100% overlap!
  
Pattern Merging:
  Merge P2 into P1
  P1 now has future connections to BOTH Z and D
```

### Pattern P1 After Merge

```
Pattern P1 (merged):
┌─────────────────────────────────────┐
│ Pattern Neuron: P1 (at level 1)    │
│ Peak Neuron: C (at level 0)        │
├─────────────────────────────────────┤
│ PAST CONNECTIONS (context):        │
│   A→B (dist=1) strength=1.0        │
│   A→C (dist=2) strength=1.0        │
│   B→C (dist=1) strength=1.0        │
├─────────────────────────────────────┤
│ FUTURE CONNECTIONS (predictions):  │
│   A→Z (dist=3) strength=1.0        │
│   B→Z (dist=2) strength=1.0        │
│   C→Z (dist=1) strength=1.0        │
│   A→D (dist=3) strength=1.0        │
│   B→D (dist=2) strength=1.0        │
│   C→D (dist=1) strength=1.0        │
└─────────────────────────────────────┘

Now P1 predicts BOTH Z and D!
```

### Episode 4: Sequence B Again (Pattern Learning!)

```
Frame 4: Z

Pattern P1 activates and predicts:
  - Z with strength=3.0 (3 connections)
  - D with strength=3.0 (3 connections)

Ground truth: Z ✓

Learning:
  ✓ Z was predicted correctly
    → Reinforce Z connections: +1.0 each
    → A→Z: 1.0 + 1.0 = 2.0
    → B→Z: 1.0 + 1.0 = 2.0
    → C→Z: 1.0 + 1.0 = 2.0
    
  ✗ D was predicted incorrectly
    → Weaken D connections: -0.1 each
    → A→D: 1.0 - 0.1 = 0.9
    → B→D: 1.0 - 0.1 = 0.9
    → C→D: 1.0 - 0.1 = 0.9
```

### Episode 5: Sequence A Again

```
Frame 4: D

Pattern P1 activates and predicts:
  - Z with strength=6.0 (3 connections * 2.0 each)
  - D with strength=2.7 (3 connections * 0.9 each)

Ground truth: D ✓

Learning:
  ✗ Z was predicted incorrectly
    → Weaken Z connections: -0.1 each
    → A→Z: 2.0 - 0.1 = 1.9
    
  ✓ D was predicted correctly
    → Reinforce D connections: +1.0 each
    → A→D: 0.9 + 1.0 = 1.9
```

### After Many Episodes

```
If Sequence A is more common (e.g., 80% A, 20% B):

Pattern P1:
┌─────────────────────────────────────┐
│ PAST CONNECTIONS:                  │
│   A→B, A→C, B→C (unchanged)        │
├─────────────────────────────────────┤
│ FUTURE CONNECTIONS:                │
│   A→Z strength=5.0  (20 episodes)  │
│   B→Z strength=5.0                 │
│   C→Z strength=5.0                 │
│   A→D strength=80.0 (80 episodes)  │
│   B→D strength=80.0                │
│   C→D strength=80.0                │
└─────────────────────────────────────┘

Predictions:
  - Z: 5.0 * 3 = 15.0
  - D: 80.0 * 3 = 240.0
  
→ D wins! (correct for 80% of cases)
```

---

## Multi-Neuron Frames Example

```
Sequence A: (A,B) → (C,D) → (E,F) → (G,H)
Sequence B: (A,B) → (C,D) → (E,F) → (I,J)

Episode 1: Learn Sequence A
  Frame 1: A, B
  Frame 2: C, D  [Learn: A→C, A→D, B→C, B→D]
  Frame 3: E, F  [Learn: C→E, C→F, D→E, D→F, ...]
  Frame 4: G, H  [Learn: E→G, E→H, F→G, F→H, ...]

Episode 2: See Sequence B
  Frame 4: I, J  [Predict: G, H ✗]
  
  Errors:
    - False positives: G, H
    - False negatives: I, J
  
  Create 2 patterns:
    - P1 for I (peak=E)
    - P2 for J (peak=F)
```

### Pattern P1 for I

```
┌─────────────────────────────────────┐
│ Pattern P1                         │
│ Peak: E                            │
├─────────────────────────────────────┤
│ PAST CONNECTIONS (all active):    │
│   A→C, A→D, B→C, B→D              │
│   C→E, C→F, D→E, D→F              │
│   A→E, A→F, B→E, B→F              │
│   (12 connections total)           │
├─────────────────────────────────────┤
│ FUTURE CONNECTIONS:                │
│   E→I, F→I, C→I, D→I, A→I, B→I    │
│   (6 connections total)            │
└─────────────────────────────────────┘
```

### Pattern P2 for J

```
┌─────────────────────────────────────┐
│ Pattern P2                         │
│ Peak: F                            │
├─────────────────────────────────────┤
│ PAST CONNECTIONS:                  │
│   Same 12 connections as P1        │
├─────────────────────────────────────┤
│ FUTURE CONNECTIONS:                │
│   E→J, F→J, C→J, D→J, A→J, B→J    │
└─────────────────────────────────────┘
```

**Key insight:** Both patterns have the same past connections (same context), but different peaks and different future connections (different predictions).

---

## Hierarchical Extension

### Level 0 → Level 1

```
Level 0: A → B → C → D → E → F → G → H → I → J
         (10 frames, exceeds baseNeuronMaxAge=5)

After frame 3, neuron A ages out (age=5)
Connection inference can only look back 4 frames
Cannot use A's context for predicting beyond frame 5

Solution: Level 1 patterns!
```

### Pattern at Level 1

```
Frame 3: Error occurs
Create Pattern P1 at level 1:
  Past: A→B, A→C, B→C
  Future: predictions
  
P1 activates at level 1 (age=0)
P1 stays active for 5 frames (baseNeuronMaxAge)

Frame 8: P1 is at age=5
But P1 represents frames 1-3!
So frame 8 can access context from frame 1
Even though frame 1 is 7 frames ago!

Effective context window: 5 + 3 = 8 frames
```

### Level 1 → Level 2

```
Level 1 patterns: P1, P2, P3, P4, P5
Each represents 3 frames of level 0

Level 2 pattern: Q1
  Past: P1→P2, P1→P3, P2→P3
  Future: predictions
  
Q1 represents 3 level-1 patterns
Each level-1 pattern represents 3 level-0 frames
Q1 represents 3*3 = 9 level-0 frames!

Effective context window grows exponentially!
```

---

## Summary Diagram

```
┌─────────────────────────────────────────────────────────┐
│                  ERROR-BASED PATTERNS                   │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Prediction Error (Surprise!)                          │
│         ↓                                               │
│  Create Pattern                                         │
│    ├─ Past Connections (context)                       │
│    └─ Future Connections (prediction)                  │
│         ↓                                               │
│  Pattern Matching                                       │
│    ├─ Same context? → Merge                            │
│    └─ Different context? → New pattern                 │
│         ↓                                               │
│  Pattern Activation                                     │
│    └─ When past connections active → predict future    │
│         ↓                                               │
│  Pattern Learning                                       │
│    ├─ Correct prediction → Reinforce (+1.0)           │
│    └─ Wrong prediction → Weaken (-0.1)                │
│         ↓                                               │
│  Forget Cycle                                           │
│    └─ Weak connections deleted                         │
│                                                         │
│  Result: Self-regulating, biologically plausible       │
│          learning system!                              │
└─────────────────────────────────────────────────────────┘
```

---

## Key Takeaways

1. **Patterns created only from errors** (surprise-based learning)
2. **Past connections = context** (when to activate)
3. **Future connections = predictions** (what to predict)
4. **Pattern merging = context discovery** (automatic)
5. **Pattern competition = ambiguity handling** (graceful)
6. **Negative reinforcement = pruning** (self-regulating)
7. **Hierarchical levels = extended context** (exponential reach)
8. **Works with multi-neuron frames** (scalable)

## Critical Design Point: Multiple Future Connections

**Patterns can and should predict multiple neurons simultaneously!**

This happens when:
- **Ambiguous context:** Same past connections lead to different outcomes (ABCD vs ABCX)
- **Multi-neuron frames:** Multiple neurons activate together (frame has A and B)

**Example: ABCD vs ABCX (50% each)**

After learning both sequences, pattern P1 has:
- Past: A→B, A→C, B→C (same for both)
- Future: A→D, B→D, C→D (strength=50) + A→X, B→X, C→X (strength=50)

At frame C, P1 predicts:
- D: 50 × 3 = 150
- X: 50 × 3 = 150

**Both D and X predicted with equal strength!** This is correct behavior for ambiguous context.

**The design is complete and ready for implementation!**
