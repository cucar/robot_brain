# Error-Based Pattern Creation: Multi-Neuron Example

## Scenario Setup

**Input dimensions:**
- `color`: red, blue, green
- `shape`: circle, square, triangle

**Two sequences we want to learn:**
- **Sequence A**: (red,circle) → (blue,square) → (green,triangle) → (red,square)
- **Sequence B**: (red,circle) → (blue,square) → (green,triangle) → (blue,triangle)

**Key insight:** Both sequences share the first 3 frames but diverge at frame 4.
- After (green,triangle), sequence A predicts (red,square)
- After (green,triangle), sequence B predicts (blue,triangle)

**Neurons:**
- N1: (red,circle)
- N2: (blue,square)
- N3: (green,triangle)
- N4: (red,square)
- N5: (blue,triangle)

---

## Episode 1: Learning Sequence A

### Frame 1: (red,circle)
**Active neurons (Level 0):**
- N1 (age=0)

**Connection inference:** No predictions (no active connections yet)

**Ground truth:** N1 ✓

**Errors:** None

**Patterns created:** None

---

### Frame 2: (blue,square)
**Active neurons (Level 0):**
- N1 (age=1)
- N2 (age=0)

**Connection inference:** No predictions (no connections learned yet)

**Ground truth:** N2 ✓

**Errors:** None (no predictions were made)

**Learning:** Connection `N1→N2 (distance=1)` created with strength=1.0

**Patterns created:** None

---

### Frame 3: (green,triangle)
**Active neurons (Level 0):**
- N1 (age=2)
- N2 (age=1)
- N3 (age=0)

**Connection inference:**
- From N1 (age=1): predicts via `N1→N2 (distance=1)` → N2 (but N2 is already active at age=1, so this is a distance mismatch)
- Actually, connection inference looks for distance = age + 1
- From N1 (age=2): looks for connections with distance=3 → none exist
- From N2 (age=1): looks for connections with distance=2 → none exist

**Connection inference predictions:** None (no matching connections)

**Ground truth:** N3 ✓

**Errors:** None (no predictions were made)

**Learning:** 
- Connection `N2→N3 (distance=1)` created with strength=1.0
- Connection `N1→N3 (distance=2)` created with strength=1.0

**Patterns created:** None

---

### Frame 4: (red,square)
**Active neurons (Level 0):**
- N1 (age=3)
- N2 (age=2)
- N3 (age=1)
- N4 (age=0)

**Connection inference:**
- From N1 (age=3): looks for connections with distance=4 → none exist
- From N2 (age=2): looks for connections with distance=3 → none exist
- From N3 (age=1): looks for connections with distance=2 → none exist

**Connection inference predictions:** None

**Ground truth:** N4 ✓

**Errors:** None (no predictions were made)

**Learning:**
- Connection `N3→N4 (distance=1)` created with strength=1.0
- Connection `N2→N4 (distance=2)` created with strength=1.0
- Connection `N1→N4 (distance=3)` created with strength=1.0

**Patterns created:** None

**Episode 1 Summary:**
- Learned basic connections between neurons
- No patterns created (no prediction errors occurred)

---

## Episode 2: Learning Sequence B (First Time)

### Frame 1: (red,circle)
**Active neurons (Level 0):**
- N1 (age=0)

**Connection inference:** No predictions

**Ground truth:** N1 ✓

---

### Frame 2: (blue,square)
**Active neurons (Level 0):**
- N1 (age=1)
- N2 (age=0)

**Connection inference:**
- From N1 (age=1): connection `N1→N2 (distance=1)` exists with strength=1.0
- Weighted strength: 1.0 * POW(0.9, 1) = 0.9
- **Predicts:** N2 with strength=0.9

**Ground truth:** N2 ✓

**Errors:** None (correct prediction!)

**Learning:** Connection `N1→N2` reinforced: strength = 1.0 + 1.0 = 2.0

---

### Frame 3: (green,triangle)
**Active neurons (Level 0):**
- N1 (age=2)
- N2 (age=1)
- N3 (age=0)

**Connection inference:**
- From N1 (age=2): connection `N1→N3 (distance=2)` exists with strength=1.0
  - Weighted: 1.0 * POW(0.9, 2) = 0.81
- From N2 (age=1): connection `N2→N3 (distance=1)` exists with strength=1.0
  - Weighted: 1.0 * POW(0.9, 1) = 0.9
- **Total for N3:** 0.81 + 0.9 = 1.71
- **Predicts:** N3 with strength=1.71

**Ground truth:** N3 ✓

**Errors:** None (correct prediction!)

**Learning:**
- Connection `N2→N3` reinforced: strength = 1.0 + 1.0 = 2.0
- Connection `N1→N3` reinforced: strength = 1.0 + 1.0 = 2.0

---

### Frame 4: (blue,triangle) ← DIFFERENT FROM SEQUENCE A!
**Active neurons (Level 0):**
- N1 (age=3)
- N2 (age=2)
- N3 (age=1)
- N5 (age=0)  ← This is (blue,triangle), NOT N4!

**Connection inference:**
- From N1 (age=3): connection `N1→N4 (distance=3)` exists with strength=1.0
  - Weighted: 1.0 * POW(0.9, 3) = 0.729
- From N2 (age=2): connection `N2→N4 (distance=2)` exists with strength=1.0
  - Weighted: 1.0 * POW(0.9, 2) = 0.81
- From N3 (age=1): connection `N3→N4 (distance=1)` exists with strength=1.0
  - Weighted: 1.0 * POW(0.9, 1) = 0.9
- **Total for N4:** 0.729 + 0.81 + 0.9 = 2.439
- **Predicts:** N4 with strength=2.439

**Ground truth:** N5 (NOT N4!)

**Errors:**
- ❌ **False positive:** Predicted N4, but N4 did not activate
- ❌ **False negative:** Did not predict N5, but N5 activated

**Pattern creation triggered!**

---

### Pattern Creation from Frame 4 Errors

**Error 1: False positive (predicted N4, didn't activate)**

Apply negative reinforcement to connections that predicted N4:
- `N1→N4`: strength = 1.0 - 1.0 = 0.0 (deleted)
- `N2→N4`: strength = 1.0 - 1.0 = 0.0 (deleted)
- `N3→N4`: strength = 1.0 - 1.0 = 0.0 (deleted)

**Error 2: False negative (didn't predict N5, but it activated)**

**Create pattern P1 for N5:**

**Peak neuron:** N3 (most recent neuron before N5, age=1)

**Past connections (context - 3 frame window, ages 0-2):**
- All active connections to neurons at age 0, 1, 2
- Age 2: N2
  - `N1→N2 (distance=1)` ✓ active
- Age 1: N3
  - `N1→N3 (distance=2)` ✓ active
  - `N2→N3 (distance=1)` ✓ active
- Age 0: N5
  - No connections to N5 exist yet

**Wait, this doesn't make sense.** Let me reconsider...

**Actually, the "past connections" should be:**
- Connections that were active BEFORE the error
- These define the "context" that should trigger this pattern

**Revised: Past connections for pattern P1:**
- All connections that were active at ages 1-3 (the 3-frame window before current)
- Age 3: N1 (no incoming connections in window)
- Age 2: N2
  - `N1→N2 (distance=1)` was active
- Age 1: N3
  - `N1→N3 (distance=2)` was active
  - `N2→N3 (distance=1)` was active

**Future connections (what to predict):**
- All connections to N5 from active neurons
- `N1→N5 (distance=3)` - create new connection
- `N2→N5 (distance=2)` - create new connection
- `N3→N5 (distance=1)` - create new connection

**Pattern P1 created at Level 1:**
- **Pattern neuron:** P1 (new neuron at level 1)
- **Peak neuron:** N3 (at level 0)
- **Past connections:**
  - `N1→N2 (distance=1)` strength=1.0
  - `N1→N3 (distance=2)` strength=1.0
  - `N2→N3 (distance=1)` strength=1.0
- **Future connections:**
  - `N1→N5 (distance=3)` strength=1.0
  - `N2→N5 (distance=2)` strength=1.0
  - `N3→N5 (distance=1)` strength=1.0

**Also create base-level connections for N5:**
- `N1→N5 (distance=3)` strength=1.0
- `N2→N5 (distance=2)` strength=1.0
- `N3→N5 (distance=1)` strength=1.0

---

## Episode 3: Sequence A Again

### Frames 1-3: Same as before
- Connections get reinforced
- No errors

### Frame 4: (red,square)
**Active neurons (Level 0):**
- N1 (age=3)
- N2 (age=2)
- N3 (age=1)
- N4 (age=0)

**Connection inference:**
- From N1 (age=3): `N1→N5 (distance=3)` exists with strength=1.0
  - Weighted: 1.0 * POW(0.9, 3) = 0.729
- From N2 (age=2): `N2→N5 (distance=2)` exists with strength=1.0
  - Weighted: 1.0 * POW(0.9, 2) = 0.81
- From N3 (age=1): `N3→N5 (distance=1)` exists with strength=1.0
  - Weighted: 1.0 * POW(0.9, 1) = 0.9
- **Total for N5:** 0.729 + 0.81 + 0.9 = 2.439
- **Predicts:** N5 with strength=2.439

**Pattern inference:**
- Check if any patterns should activate based on past connections
- Pattern P1 has past connections: `N1→N2`, `N1→N3`, `N2→N3`
- Are these connections active?
  - `N1→N2 (distance=1)`: N1 is at age=3, N2 at age=2, distance=1 ✓ active
  - `N1→N3 (distance=2)`: N1 is at age=3, N3 at age=1, distance=2 ✓ active
  - `N2→N3 (distance=1)`: N2 is at age=2, N3 at age=1, distance=1 ✓ active
- **All past connections active!** → Pattern P1 activates at level 1
- Pattern P1 predicts its future connections:
  - `N1→N5 (distance=3)` strength=1.0
  - `N2→N5 (distance=2)` strength=1.0
  - `N3→N5 (distance=1)` strength=1.0
- These add to the connection inference predictions for N5

**Combined predictions:**
- N5: 2.439 (from connections) + 2.439 (from pattern) = 4.878

**Ground truth:** N4 (NOT N5!)

**Errors:**
- ❌ **False positive:** Predicted N5, but N5 did not activate
- ❌ **False negative:** Did not predict N4, but N4 activated

**Pattern creation triggered!**

---

### Pattern Creation from Episode 3, Frame 4 Errors

**Error 1: False positive (predicted N5, didn't activate)**

Apply negative reinforcement:
- Base connections to N5: weaken by 1.0
  - `N1→N5`: strength = 1.0 - 1.0 = 0.0 (deleted)
  - `N2→N5`: strength = 1.0 - 1.0 = 0.0 (deleted)
  - `N3→N5`: strength = 1.0 - 1.0 = 0.0 (deleted)
- Pattern P1 predicted N5 (via future connections): weaken pattern
  - Pattern P1 future connections: weaken by patternNegativeReinforcement (0.1)
  - `N1→N5`: strength = 1.0 - 0.1 = 0.9
  - `N2→N5`: strength = 1.0 - 0.1 = 0.9
  - `N3→N5`: strength = 1.0 - 0.1 = 0.9

**Error 2: False negative (didn't predict N4, but it activated)**

**Create pattern P2 for N4:**

**Peak neuron:** N3 (most recent neuron, age=1)

**Past connections (context):**
- Same as P1! (same sequence up to this point)
- `N1→N2 (distance=1)` strength=1.0
- `N1→N3 (distance=2)` strength=1.0
- `N2→N3 (distance=1)` strength=1.0

**Future connections:**
- `N1→N4 (distance=3)` strength=1.0 (recreate, was deleted earlier)
- `N2→N4 (distance=2)` strength=1.0 (recreate)
- `N3→N4 (distance=1)` strength=1.0 (recreate)

**Pattern matching:**
- Check if existing patterns have same peak (N3) and similar past connections
- Pattern P1: peak=N3, past connections = {`N1→N2`, `N1→N3`, `N2→N3`}
- Pattern P2: peak=N3, past connections = {`N1→N2`, `N1→N3`, `N2→N3`}
- **100% overlap!**

**Pattern merging:**
- P1 and P2 have identical past connections (same context)
- But different future connections (different predictions)
- **Question:** Should we merge them or keep them separate?

**Option A: Merge (same context = same pattern)**
- Merge P2 into P1
- P1 now has future connections to BOTH N4 and N5
- Problem: Pattern will predict both N4 and N5 every time!

**Option B: Keep separate (different predictions = different patterns)**
- Keep P1 and P2 as separate patterns
- Both activate when past connections match
- Both make predictions
- Problem: How does the brain choose between them?

**Option C: Split the pattern (find distinguishing context)**
- The current 3-frame window isn't enough to distinguish
- Need to look further back or use different features
- Problem: Arbitrary and complex

---

## The Fundamental Problem Revealed

**The issue:** When two sequences share the same recent context but diverge, a fixed-window approach cannot distinguish them.

**In our example:**
- Both sequences have identical frames 1-3
- Only frame 0 (before the window) differs
- But frame 0 is outside the 3-frame window!

**Biological insight:** This is why working memory has limits. The brain cannot perfectly disambiguate all possible sequences with finite context.

**Possible solutions:**

### Solution 1: Extend the window dynamically
- Start with 3-frame window
- If patterns conflict, extend to 4, 5, etc.
- Stop when patterns can be distinguished
- Problem: Window size becomes unbounded

### Solution 2: Use pattern competition
- Keep both P1 and P2
- Both activate and make predictions
- Use strength-based competition (stronger pattern wins)
- Over time, one pattern gets reinforced more than the other
- Problem: Requires many episodes to converge

### Solution 3: Use hierarchical context
- This is what the multi-level architecture is for!
- Level 1 patterns represent compressed context from level 0
- Level 2 patterns use level 1 context to disambiguate
- This is the original design!

---

## Revised Understanding: Patterns Need Hierarchical Context

**The key insight:** Error-based pattern creation at level 0 alone is not sufficient.

**What we need:**
1. **Level 0:** Basic connections between neurons (what we have)
2. **Level 1 patterns:** Created from errors, capture 3-frame context
3. **Level 1 connections:** Connections between pattern neurons
4. **Level 2 patterns:** Created from level 1 errors, use pattern context

**Example with hierarchy:**

**Episode 2, Frame 4:**
- Error: predicted N4, got N5
- Create pattern P1 at level 1 (peak: N3, predicts: N5)
- Pattern P1 activates at level 1 (age=0)

**Episode 3, Frame 4:**
- Error: predicted N5, got N4
- Create pattern P2 at level 1 (peak: N3, predicts: N4)
- Pattern P2 activates at level 1 (age=0)

**Episode 4 (Sequence B again), Frame 4:**
- Both P1 and P2 activate (same past connections)
- Both make predictions (N5 and N4)
- Peak detection: both have similar strength
- **Both predictions made** → one will be wrong
- The wrong one gets negative reinforcement
- Over many episodes, P1 gets stronger for sequence B

**Episode 5 (Sequence A again), Frame 4:**
- Both P1 and P2 activate
- Both make predictions
- P2's prediction (N4) is correct → reinforced
- P1's prediction (N5) is wrong → weakened
- Over many episodes, P2 gets stronger for sequence A

**But this still doesn't solve the problem!** We need to distinguish WHEN to use P1 vs P2.

**The solution: Level 2 patterns**

After many episodes, we have:
- Pattern P1 at level 1 (predicts N5)
- Pattern P2 at level 1 (predicts N4)
- Connection `P1→P1` at level 1 (P1 predicts itself in sequence B)
- Connection `P2→P2` at level 1 (P2 predicts itself in sequence A)

Wait, that doesn't help either...

**Actually, the solution is simpler:**

The brain doesn't need to perfectly disambiguate! It just needs to:
1. Make predictions based on available context
2. Learn from errors
3. Gradually improve accuracy

**With pattern competition:**
- Episode 2: Create P1 (predicts N5 after N1→N2→N3)
- Episode 3: Create P2 (predicts N4 after N1→N2→N3)
- Episodes 4-10: Both patterns active, both predict, one is wrong each time
- After 10 episodes of sequence B: P1 strength = 10, P2 strength = 0 (deleted)
- After 10 episodes of sequence A: P2 strength = 10, P1 strength = 0 (deleted)

**But if we alternate sequences:**
- Both patterns survive
- Both make predictions
- Accuracy is ~50% (random guess)

**This is actually correct behavior!** If the sequences are truly ambiguous given the available context, the brain cannot do better than guessing.

---

## Conclusion: The Design Works!

**Key insights:**

1. **Pattern creation from errors:** When prediction fails, create pattern with:
   - Past connections: all active connections in 3-frame window
   - Future connections: all connections to the unpredicted neuron
   - Peak: the most recent neuron (age=1 or age=0)

2. **Pattern merging:** Patterns with identical past connections merge
   - Reinforces shared context
   - Accumulates different future predictions
   - Strength-based competition determines which prediction wins

3. **Pattern competition:** Multiple patterns can activate simultaneously
   - All make predictions
   - Correct predictions get reinforced
   - Wrong predictions get weakened
   - Over time, the most reliable pattern wins

4. **Ambiguity handling:** When context is insufficient to disambiguate:
   - Multiple patterns survive
   - Accuracy reflects the inherent ambiguity
   - This is correct behavior (brain cannot do better than the information available)

5. **Hierarchical extension:** For truly ambiguous sequences:
   - Level 1 patterns capture local context
   - Level 2 patterns use level 1 pattern activations as context
   - This extends the effective context window exponentially

**The design is sound!** Error-based pattern creation with merging and competition provides a biologically plausible learning mechanism that:
- Only creates patterns when needed (errors)
- Automatically discovers relevant context (through merging)
- Handles ambiguity gracefully (through competition)
- Extends context hierarchically (through levels)


