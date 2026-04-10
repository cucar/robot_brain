# Unified Pattern Recognition — Implementation Plan

## Problem

`matchPatterns` calls each recognizer neuron separately per age. The same parent neuron active at ages 0, 1, and 2 gets called three times with three different contexts. A `recognizedPatterns` Set prevents duplicate pattern activations, but the neuron still runs `matchPattern` + `refineContext` multiple times — wasting work and injecting noise (refining against non-winning contexts).

This violates two principles:
1. **Biological**: a neuron fires once per recognition cycle, not once per age
2. **MPI-ready**: each neuron call is a potential network message; one call per neuron per frame is the target (see rust-migration-plan.md Phase 1)

## Goal

Each recognizer neuron is called **once** with everything it needs. It returns a result. The thalamus handles all cross-neuron side effects. The `recognizedPatterns` duplicate check becomes unnecessary and is removed.

---

## Architecture

### Current Flow

```
Thalamus.recognizeLevel(level)
  → getContextByAge()        → Map<age, Context>     (one Context per age)
  → getRecognizers()          → [{neuron, age, context}, ...]  (same neuron appears N times)
  → matchPatterns()           → for each entry:
                                  neuron.matchPattern(context, frameNumber, neurons)
                                    ↳ scores patterns
                                    ↳ refineContext (mutates routing table)
                                    ↳ updates contextRefs on OTHER neurons  ← cross-neuron access
                                  if match && !recognizedPatterns.has(match.pattern)  ← duplicate guard
```

### Target Flow

```
Thalamus.recognizeLevel(level)
  → buildLevelContext()       → Context              (one universal context for the level)
  → getRecognizers()          → [{neuron, activeAges, alivePatternIds}, ...]  (one entry per neuron)
  → matchPatterns()           → for each entry:
                                  neuron.matchPattern(levelContext, activeAges, alivePatternIds)
                                    ↳ scores patterns across all active ages
                                    ↳ refines routing table locally
                                    ↳ returns {patternId, age, score, novel, removedRefs}
                                  thalamus updates contextRefs based on returned novel/removedRefs
```

---

## Changes by File

### 1. `context.js` — Add `matchWithOffset`

New method on Context. Like `match()`, but shifts all known-context distances by an offset before comparing against observed.

A pattern stored with `(neuronX, distance=3)` was learned when the parent was at some age. If the parent is now at age 2, the same neuronX (if at absolute age 5) appears in the level context at distance 5. The offset is `age`: look for `distance + offset` in the observed context.

```javascript
matchWithOffset(observed, offset, mergeThreshold) {
    // Same logic as match(), but when checking observed distances:
    // - common:  observedDistances.has(distance + offset)
    // - scoring: getMatchScore(strength, distance + offset, observedDistances)
    // - novel:   iterate observed, subtract offset to get pattern-relative distance
    //            skip entries where (distance - offset) < 1 (before the parent)
}
```

**Key detail**: `common`, `missing`, `novel` entries store the **original pattern-relative distances** (not offset), because refinement operates on the routing table which uses pattern-relative distances.

For novel entries, the pattern-relative distance is `observedDistance - offset`. Only include novel entries where this is ≥ 1 (context neurons must be older than the parent).

### 2. `neuron.js` — Refactor `matchPattern`

**Remove** the `neurons` parameter. The neuron no longer accesses other neurons.

**New signature**:
```javascript
matchPattern(levelContext, activeAges, alivePatternIds)
```

**Logic**:
```javascript
matchPattern(levelContext, activeAges, alivePatternIds) {
    // 1. Score: try each pattern × each active age, keep best
    let best = null;
    for (const [patternId, patternContext] of this.routingTable) {
        if (!alivePatternIds.has(patternId)) continue;
        for (const age of activeAges) {
            const match = patternContext.matchWithOffset(levelContext, age, this.mergeThreshold);
            if (match && (!best || match.score > best.score)) {
                match.patternId = patternId;
                match.age = age;
                best = match;
            }
        }
    }
    if (!best) return null;

    // 2. Refine locally — strengthen, add, weaken (all local routing table ops)
    const context = this.routingTable.get(best.patternId);
    for (const entry of best.common) context.strengthenNeuron(entry.neuronId, entry.distance);
    for (const entry of best.novel) this.addContext(best.patternId, entry.neuronId, entry.distance, 1);

    // 3. Weaken missing — collect removed refs for thalamus
    const removedRefs = [];
    for (const entry of best.missing) {
        const canDelete = context.weakenNeuron(entry.neuronId, entry.distance);
        if (canDelete) {
            context.remove(entry.neuronId, entry.distance);
            // Check if any other pattern in this parent still references this neuron at this distance
            let referenced = false;
            for (const [otherId, otherCtx] of this.routingTable) {
                if (otherId !== best.patternId && otherCtx.hasKey(entry.neuronId, entry.distance)) {
                    referenced = true;
                    break;
                }
            }
            if (!referenced) removedRefs.push({ neuronId: entry.neuronId, distance: entry.distance });
        }
    }
    best.removedRefs = removedRefs;
    return best;
}
```

**Delete** `refineContext` — its logic is now inline in `matchPattern`.

### 3. `thalamus.js` — Refactor recognition pipeline

#### `buildLevelContext(level, activeNeuronsByAge)` (replaces `getContextByAge`)

Build one Context containing all active same-level neurons with distances relative to age 0 (i.e., distance = absolute age). Every neuronId gets every distance it appears at.

```javascript
buildLevelContext(level, activeNeuronsByAge) {
    const context = new Context();
    for (let ctxAge = 1; ctxAge < activeNeuronsByAge.length; ctxAge++)
        for (const neuronId of (activeNeuronsByAge[ctxAge] ?? new Map()).keys()) {
            if (this.skipActionNeuron(...) || this.neuronLevels.get(neuronId) !== level) continue;
            context.addNeuron(neuronId, ctxAge, 1);  // distance = absolute age
        }
    return context;
}
```

No per-age splitting. One context for the entire level.

#### `getRecognizers(level, activeNeuronsByAge)` (refactored)

Returns one entry per neuron with all its active ages and its alive pattern IDs.

```javascript
getRecognizers(level, activeNeuronsByAge) {
    const recognizerMap = new Map(); // neuronId → {neuron, activeAges, alivePatternIds}
    for (let age = 0; age < activeNeuronsByAge.length; age++) {
        for (const [neuronId, state] of (activeNeuronsByAge[age] ?? new Map())) {
            if (state.activatedPattern !== null) continue;
            const neuron = this.neurons.get(neuronId);
            if (!neuron || this.skipActionNeuron(neuron) || this.neuronLevels.get(neuronId) !== level) continue;
            if (!recognizerMap.has(neuronId))
                recognizerMap.set(neuronId, { neuron, activeAges: [], alivePatternIds: null });
            recognizerMap.get(neuronId).activeAges.push(age);
        }
    }

    // Build alivePatternIds for each recognizer (pre-filter forgotten patterns)
    for (const entry of recognizerMap.values()) {
        const alive = new Set();
        for (const [patternId] of entry.neuron.routingTable) {
            const pattern = this.neurons.get(patternId);
            if (pattern && pattern.getEffectiveActivationStrength(this.currentFrame) > 0)
                alive.add(patternId);
        }
        entry.alivePatternIds = alive;
    }

    return recognizerMap;
}
```

Note: `currentFrame` needs to be passed in or stored. Currently it's passed as `frameNumber` to `matchPatterns` — same thing, just threaded through.

#### `matchPatterns(recognizerMap, levelContext)` (refactored)

```javascript
matchPatterns(recognizerMap, levelContext) {
    const matchedPatterns = [];
    for (const { neuron: parent, activeAges, alivePatternIds } of recognizerMap.values()) {
        const match = parent.matchPattern(levelContext, activeAges, alivePatternIds);
        if (!match) continue;

        // Cross-neuron side effects: thalamus handles contextRef updates
        for (const entry of match.novel)
            this.neurons.get(entry.neuronId)?.addContextRef(parent.id, entry.distance);
        for (const ref of match.removedRefs)
            this.neurons.get(ref.neuronId)?.removeContextRef(parent.id, ref.distance);

        // Resolve pattern neuron for downstream (activatePattern needs the Neuron object)
        const pattern = this.neurons.get(match.patternId);
        match.pattern = pattern;

        matchedPatterns.push({ parent, age: match.age, match });
    }
    return matchedPatterns;
}
```

No `recognizedPatterns` Set. Each neuron appears once → each pattern can only be matched once.

#### `recognizeLevel` (updated)

```javascript
recognizeLevel(level, activeNeuronsByAge, frameNumber) {
    const levelContext = this.buildLevelContext(level, activeNeuronsByAge);
    if (levelContext.size === 0) return [];

    const recognizers = this.getRecognizers(level, activeNeuronsByAge);
    if (recognizers.size === 0) return [];

    return this.matchPatterns(recognizers, levelContext);
}
```

### 4. `brain.js` — No changes expected

`recognizeLevel` returns the same structure: `[{parent, age, match}]`. The `match` object still has `.pattern` (Neuron), `.score`, etc. `activatePattern` and `registerDeath` work unchanged.

---

## `matchWithOffset` — Detailed Design

The core new method. Replaces `context.match()` for recognition. The known context stores pattern-relative distances. The observed (level) context stores absolute ages as distances. The offset converts between them: `absoluteAge = patternDistance + offset`.

```javascript
matchWithOffset(observed, offset, mergeThreshold) {
    const common = [];
    const missing = [];
    let totalCount = 0;
    let score = 0;

    for (const [neuronId, distanceMap] of this.entries) {
        const observedDistances = observed.entries.get(neuronId);
        for (const [distance, strength] of distanceMap) {
            if (strength <= 0) continue;
            totalCount++;

            const absoluteDistance = distance + offset;
            if (observedDistances?.has(absoluteDistance))
                common.push({ neuronId, distance, strength });
            else
                missing.push({ neuronId, distance, strength });

            score += this.getMatchScore(strength, absoluteDistance, observedDistances);
        }
    }

    if (totalCount === 0) return null;
    if (common.length / totalCount < mergeThreshold) return null;

    // Novel entries: in observed but not in known (with offset adjustment)
    const novel = [];
    for (const [neuronId, distanceMap] of observed.entries) {
        const knownDistances = this.entries.get(neuronId);
        for (const [absoluteDistance, strength] of distanceMap) {
            const patternDistance = absoluteDistance - offset;
            if (patternDistance < 1) continue;  // must be older than parent
            if (!knownDistances || !knownDistances.has(patternDistance) || knownDistances.get(patternDistance) <= 0)
                if (!this.hasPartialMatch(patternDistance, knownDistances)) {
                    novel.push({ neuronId, distance: patternDistance, strength });
                    score -= strength;
                }
        }
    }

    score = Math.round(score * 1e14) / 1e14;
    return { score, common, missing, novel };
}
```

**Important**: `getMatchScore` receives `absoluteDistance` and `observedDistances` (which are also in absolute distances). This works correctly because both are in the same reference frame.

---

## Implementation Steps

Each step is self-contained. Run ALL tests after each step. Results must be identical (or improve).

### Step 1 — Add `matchWithOffset` to Context

- Add the new method to `context.js`
- Unit test: create a known context and observed context, verify `matchWithOffset` with offset=0 produces identical results to `match`
- Unit test: verify offset>0 correctly shifts distances

### Step 2 — Refactor `neuron.matchPattern`

- New signature: `matchPattern(levelContext, activeAges, alivePatternIds)`
- Inline the refinement logic (strengthen/add/weaken)
- Return `removedRefs` for thalamus
- Remove `refineContext` method
- Do NOT change callers yet — this will break the build temporarily

### Step 3 — Refactor thalamus recognition pipeline

- Replace `getContextByAge` with `buildLevelContext`
- Refactor `getRecognizers` to group by neuron, collect activeAges, build alivePatternIds
- Refactor `matchPatterns` to call each neuron once, handle contextRef side effects
- Update `recognizeLevel` to use the new flow
- Remove `recognizedPatterns` Set

### Step 4 — Verify

- Run existing test suite — results must be identical or better
- Run stock accuracy benchmark — compare accuracy and ROI
- If accuracy changes, investigate which patterns are now matched differently

---

## What This Enables

1. **Removes duplicate check** — `recognizedPatterns` Set is gone, the code is cleaner
2. **Each neuron called once** — ready for MPI where each call is a network message
3. **No cross-neuron access in Neuron** — neurons are pure data processors, thalamus routes
4. **Better refinement** — context is refined once against the winning match, not polluted by non-winning age perspectives
5. **Aligns with rust-migration-plan.md Phase 1** — moves toward `neuron.processFrame()` single-call model

## Risks

- **Accuracy change**: matching with offset instead of per-age contexts changes which patterns match. The partial-credit scoring in `getMatchScore` now operates on absolute distances, which may score differently. Benchmark carefully.
- **Novel entry filtering**: the `patternDistance < 1` guard for novel entries needs verification — ensures we don't add context entries for neurons that are younger than or same age as the parent.
- **`refineContext` callers**: verify no other code path calls `refineContext` before deleting it.