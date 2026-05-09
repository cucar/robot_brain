# Experiment: Exponential Temporal Binning in Cortex

## Motivation

Higher-level patterns live exponentially longer (decay rate = `baseRate / contextLength^(level-1)`), but store context connections at exact frame distances. This is a mismatch: a level-4 pattern distinguishes distance 47 from 48, which is meaningless at its timescale, while being unable to represent relationships at distance 500, which is well within its lifespan.

Exponential time bins fix this by giving every level the same number of bins but scaling bin width with level. This mirrors the hippocampal exponential bin scheme for moment action connections — cortex and hippocampus would speak the same temporal language.

## Bin scheme

Bin k for a pattern at level N covers distances `[contextLength^(k-1), contextLength^k)`.

With contextLength=5 and 10 bins:

| Bin | Level 1 (frames) | Level 2 (frames) | Level 3 (frames) |
|-----|-------------------|-------------------|-------------------|
| 0   | 1                 | 1-4               | 1-24              |
| 1   | 2-4               | 5-24              | 25-124             |
| 2   | 5-24              | 25-124             | 125-624            |
| ... | ...               | ...               | ...               |

Level 1 bins are narrow enough to approximate frame-resolution (bin 0 = distance 1 only). Higher levels get progressively coarser. Total horizon at level N with B bins = `contextLength^(N + B - 2)` frames, which naturally tracks the pattern's lifespan.

General formula:

```
distanceToBin(distance, level, contextLength, numBins):
    if distance <= 0: return 0
    // shift so level-1 bin 0 = distance 1
    effectiveDistance = distance * contextLength^(level - 1)
    bin = floor(log(effectiveDistance) / log(contextLength))
    return clamp(bin, 0, numBins - 1)
```

Note: level 1 is a special case of this — bins are narrow enough that low distances get distinct bins, preserving near-frame-resolution for the lowest level.

## Changes required

### 1. Context class (`brain/src/context.js`)

Currently: `Map<neuronId, Map<distance, strength>>` where distance is an exact frame count.

After: `Map<neuronId, Map<bin, strength>>` where bin is a small integer (0 to numBins-1).

- `addNeuron(neuronId, distance, strength)` — convert distance to bin before storing. If the bin already has an entry for this neuron, merge (add strengths) instead of throwing.
- `find(neuronId, distance)` — convert distance to bin, then look up.
- `strengthenNeuron(neuronId, distance)` — convert distance to bin.
- `weakenNeuron(neuronId, distance)` — convert distance to bin.
- `getMatchScore(strength, distance, observedDistances)` — both sides use bins. Exact match = full strength. Bin mismatch penalty uses bin delta (not frame delta), giving automatic coarser tolerance at higher levels.
- `match(observed, offset, mergeThreshold, excludeIds)` — the offset/absoluteDistance logic needs rethinking. Currently converts pattern-relative distance to absolute by adding offset. With bins, the offset must be converted to a bin shift. This is the trickiest part — see open questions.

Context needs to know level and contextLength to do the conversion. Options:
- Pass them into the constructor and store them (context becomes level-aware).
- Pass them into each method that needs conversion (keeps context generic but verbose).

Recommendation: constructor params. A context is always owned by a specific neuron at a specific level.

### 2. Neuron / routing table creation (`brain/src/thalamus.js`)

When a pattern is created via `recognizePatterns`, its context entries are built from observed context neurons with exact distances. These distances must be converted to bins at the pattern's level before storing.

Key site: `createPatternNeuron` (or equivalent) where context entries are assembled from the recognition result's common/novel entries.

### 3. Pattern matching (`context.match`)

The offset parameter (parent's active age) shifts pattern-relative distances to absolute for comparison against observed context. With bins:

- The observed context arrives with exact frame distances, binned at the observer's level.
- The known context has bins at the pattern's level.
- The offset (parent age) must be factored into binning, not added as a frame count.

Approach: when matching, convert the observed absolute distance to the pattern's bin scheme using `distanceToBin(absoluteDistance - parentAge, level, contextLength)` on the fly, rather than pre-binning the observed context.

### 4. Voting (`neuron.vote` or equivalent)

Votes currently carry exact distance. With bins, votes carry bin index. Downstream consumers that use distance for time-decay weighting should use the bin's representative distance (e.g., geometric mean of bin boundaries) instead.

### 5. Connection learning in `recognizePatterns`

When a pattern fires and the context is updated (strengthen common, weaken missing, add novel), the distance-to-bin conversion must be applied. Strengthening now targets a bin, so multiple exact-distance observations that fall in the same bin reinforce the same entry.

## Migration / backward compatibility

This changes the serialization format of context entries (bin index vs exact distance). Existing persisted neurons would need migration or a version flag. For the experiment phase, simplest to start fresh rather than migrate.

## Open questions

1. **Level-1 bin resolution.** With contextLength=5, level-1 bin 0 covers only distance=1, bin 1 covers 2-4, bin 2 covers 5-24. That's already coarser than the current frame-exact storage for distances > 1. Is that acceptable, or should level 1 use a separate linear scheme (preserving current behavior) and only switch to exponential at level 2+?

2. **Soft bin assignment.** The hippocampus design uses a Gaussian kernel over log-dt for soft bin assignment. Should cortex do the same (distribute strength across adjacent bins) or is hard assignment sufficient given the error-driven learning that already fuzzes matching?

3. **Offset handling in match.** The current offset arithmetic is clean because distances are integers and addition is exact. With bins, there's quantization: observed distance 47 with offset 10 = effective distance 37, which might land in a different bin than if the pattern had originally seen distance 37. Need to decide whether this quantization error matters or is actually the right behavior (coarser tolerance at higher levels).

4. **Context window length.** Currently the context window is a fixed number of frames. With exponential bins, the effective context window at higher levels spans far more frames. Does the brain need to feed longer history to higher-level pattern recognition, or is the existing window sufficient because higher-level patterns fire less frequently anyway?

5. **numBins parameter.** Should this be global (same for all levels) or per-level? Global is simpler and means every pattern has the same memory footprint regardless of level.

## Validation

- Level-1 patterns should behave nearly identically to current (regression test).
- Higher-level patterns should form with fewer, coarser context entries.
- Higher-level patterns should be able to represent temporal relationships beyond the current context window.
- Total memory usage for context storage should decrease (fewer entries per high-level pattern).
- Prediction accuracy on existing benchmarks should not regress for level-1, and may improve for higher levels.

## Relationship to hippocampus design

This makes cortical patterns and hippocampal moments use the same temporal binning principle:
- Patterns: exponential bins for **context connections** (what was happening when this pattern applies).
- Moments: exponential bins for **action connections** (what to do at various time horizons).

When moments age into classes and become structurally indistinguishable from patterns, their bin schemes should be compatible. A moment's action bins already use the exponential scheme; if its context connections also used exponential bins (inherited from the patterns it was unioned from), the two systems are fully unified.
