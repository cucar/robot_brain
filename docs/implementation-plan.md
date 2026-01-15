# Implementation Plan: Architecture vs Current Code

This document identifies deviations between the brain architecture design and the current implementation.

---

## ✅ Correctly Implemented

### 1. Two-Hierarchy Structure
- Events and actions properly separated with different learning signals
- Events use strength-based voting, actions use reward-based Boltzmann selection
- Action neurons can never be sources (enforced in `reinforceConnections`)

### 2. Pattern Override Rule
- Pattern inference correctly overrides connection inference
- Implemented in `aggregateVotesByTarget()` and `populateInferenceSources()`

### 3. Voting Architecture
- Level weighting: `POW(levelVoteMultiplier, level)` ✓
- Distance weighting: `POW(peakTimeDecayFactor, distance-1)` ✓
- Boltzmann selection for actions, deterministic for events ✓

### 4. Pattern_past Merge Logic
- Common connections strengthened, novel added, missing weakened ✓
- Implemented in `mergePatternPast()`

### 5. Pattern_future Merge for Events
- Positive/negative reinforcement and novel connection addition ✓
- Implemented in `mergePatternFuture()`

### 6. Error Pattern Creation
- Created when connection inference fails with high confidence ✓
- Peak must have context and no active pattern ✓

### 7. Exploration
- Probability inversely proportional to inference strength ✓
- Never stops completely (minExploration floor) ✓

### 8. Reward Application
- Exponential smoothing, channel-specific credit assignment ✓

---

## ⚠️ Deviations Needing Clarification

### 1. Action Vote Weight Calculation
**Architecture**: Vote weight is `reward * level_weight` (strength doesn't matter for actions).

**Current code**: `collectVotes()` uses `strength * distance_decay * level_weight`, then Boltzmann selects on `reward`.

**Question**: Should strength be excluded from action vote collection entirely?

### 2. Action Pattern Exploration
**Architecture**: Pattern finds ALL valid actions and adds them to pattern_future, then Boltzmann selects.

**Current code**: Adds ONE alternative per frame (incremental exploration).

**Status**: Acceptable - achieves same goal over multiple frames.

---

## ⚠️ Needs Verification

### 1. Pattern-to-Pattern Connection Inference
**Architecture**: Patterns can form connections to other patterns, enabling sequences longer than baseNeuronMaxAge.

**Current code**: Infrastructure exists but commented debug code suggests uncertainty:
```javascript
// const level1PlusConnectionVotes = connectionVotes.filter(v => v.from_type === null);
```

**Recommendation**: Create tests verifying level 1+ patterns can predict each other via connections.

---

## 🔧 Recommended Actions

| Priority | Action |
|----------|--------|
| 1 | Verify pattern-to-pattern inference with dedicated test |
| 2 | Clarify if action votes should exclude strength |
| 3 | Add diagnostic output for pattern hierarchy |

---

## Summary

The implementation is **largely aligned** with the architecture. Main areas:
- Pattern-to-pattern inference needs verification
- Action vote weight calculation needs clarification

