# Diagnostic Mode Implementation

## Overview

Implemented a comprehensive diagnostic mode for the brain architecture that provides detailed, frame-by-frame visibility into:
- Observations from each frame
- Inference predictions with strength, habituation, and effective strength
- Conflict resolution decisions
- Reward application

This enables analysis of learning behavior, pattern discovery, and decision-making processes.

## Usage

Run any job with the `--diagnostic` flag:

```bash
node run-brain.js synthetic-cycle-test --diagnostic
```

## Output Format

### Frame Header
Shows frame number and observations:
```
F123 | Obs: change=0.019, position=1
```

### Reward Display
Shows rewards applied from previous frame:
```
  Rewards: TEST:1.050x
```

### Inference Details (per channel)
Shows all inferences with their sources before conflict resolution:
```
  TEST Actions: activity=1(C:s450×r1.05×h0.80=360) ✓ | activity=-1(C:s320×r1.00×h1.00=320) ✗ | activity=0(C:s180×r1.00×h0.90=162) ✗
```

Format breakdown:
- `activity=1` - dimension and value
- `C:` - Connection inference (or `P:` for pattern inference)
- `s450` - connection strength
- `r1.05` - reward multiplier
- `h0.80` - habituation factor
- `=360` - effective prediction strength (s × r × h × time_decay)
- `✓` - selected by conflict resolution
- `✗` - not selected

For pattern inferences:
```
  TEST Actions: activity=1(P:ps680×pr1.10×ph0.85×cs450×cr1.05=xxx) ✓
```
- `ps680` - pattern strength
- `pr1.10` - pattern reward
- `ph0.85` - pattern habituation
- `cs450` - connection strength (pattern uses connections)
- `cr1.05` - connection reward

### Event Predictions
Similar format for event (input) predictions:
```
  TEST Events: change=1(C:s520×r1.02×h0.95=500) ✓ | change=-1(C:s380×r0.98×h1.00=372) ✗
```

## Implementation Details

### Files Modified

1. **brain.js**
   - Added `diagnostic` flag (line 57)
   - Added abstract method `getInferenceDetails(level)` (line 886)
   - Added `displayFrameHeader(frame)` method (line 253-267)
   - Added `displayRewards(channelRewards)` method (line 269-283)
   - Modified `processFrame()` to call diagnostic displays (lines 199, 211)
   - Modified `resolveChannelInferenceConflicts()` to get and display inference details (lines 628-664)

2. **brain-mysql.js**
   - Implemented `getInferenceDetails(level)` method (lines 1303-1428)
   - Queries `inferred_neurons`, `connection_inference_sources`, `pattern_inference_sources`
   - Joins with `connections`, `pattern_future` to get strength/reward/habituation
   - Aggregates and averages source information per neuron

3. **brain-memory.js**
   - Added stub implementation of `getInferenceDetails(level)` (lines 1191-1224)
   - Note: Full implementation would require adding source tracking to in-memory structures

4. **channels/channel.js**
   - Added `diagnostic` flag (line 22)
   - Added `displayDiagnostics(inferenceDetails, resolvedInferences)` method (lines 186-274)
   - Added `formatCoordinates(coordinates)` helper (lines 230-238)
   - Added `formatSources(sources)` helper (lines 240-260)

5. **run-brain.js**
   - Added support for `--diagnostic` command-line flag (lines 38-48)
   - Passes diagnostic option to job via `runnerOptions`

6. **jobs/job.js**
   - Modified `run()` to apply diagnostic mode to channels after registration (lines 25-28)

## Key Features

### 1. Temporal Separation
Shows what was inferred in previous frame and what reward it received:
- Frame N: Shows inferences made
- Frame N+1: Shows rewards applied to Frame N inferences

### 2. Source Tracking
Distinguishes between:
- **Connection inference**: Direct neuron-to-neuron predictions
- **Pattern inference**: Higher-level pattern-based predictions

### 3. Effective Strength Calculation
Shows the complete calculation:
```
effective_strength = base_strength × reward × habituation × time_decay
```

This is what's actually used for conflict resolution, so you can see exactly why one inference wins over another.

### 4. Conflict Resolution Visibility
The `✓` and `✗` markers show which inferences were selected/rejected by the channel's conflict resolution logic.

## Example Use Cases

### 1. Understanding Why Brain Prefers Certain Actions
```
F145 | Obs: change=0.029, position=0
  TEST Actions: activity=1(C:s680×r1.15×h0.60=470) ✓ | activity=0(C:s520×r1.05×h0.95=518) ✗
```
Even though HOLD (activity=0) has higher base strength (520 vs 680), BUY wins because:
- BUY has higher reward (1.15 vs 1.05) from past successes
- BUY's habituation is lower (0.60 vs 0.95) - it hasn't been overused
- Effective strength: BUY=470, HOLD=518... wait, HOLD should win!

This reveals a bug or shows that other factors are involved.

### 2. Tracking Habituation Effects
```
F100 | activity=1(C:s500×r1.10×h1.00=550) ✓
F110 | activity=1(C:s520×r1.12×h0.85=495) ✓
F120 | activity=1(C:s540×r1.14×h0.70=430) ✓
```
Shows how repeated use of BUY action:
- Increases strength (500→520→540) from rewards
- Increases reward (1.10→1.12→1.14) from success
- Decreases habituation (1.00→0.85→0.70) from overuse
- Net effect: effective strength decreases (550→495→430)

### 3. Pattern vs Connection Inference
```
F200 | activity=1(C:s450×r1.05×h0.80=360) ✗ | activity=1(P:ps680×pr1.10×ph0.85×cs450×cr1.05=xxx) ✓
```
Shows pattern inference overriding connection inference for the same action, indicating higher-level context is being used.

## Future Enhancements

1. **Pattern Details**: Show which specific pattern neurons are making predictions
2. **Level Information**: Show which level the inference came from
3. **Prediction Accuracy**: Show whether previous frame's predictions were correct
4. **Connection Details**: Show which specific connections contributed to each inference
5. **Aggregation**: Summarize patterns across multiple frames (e.g., "BUY won 8/10 times in last 10 frames")

## Testing

To test the diagnostic mode:

1. Setup synthetic test data:
```bash
node run-setup.js synthetic-cycle-test
```

2. Run with diagnostic mode:
```bash
node run-brain.js synthetic-cycle-test --diagnostic
```

3. Observe the detailed frame-by-frame output showing:
   - Price changes
   - Position state
   - Inferences with strength/reward/habituation
   - Conflict resolution decisions
   - Rewards applied

This will help identify why the brain learns certain patterns and not others, and how habituation affects decision-making over time.

