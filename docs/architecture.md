# Brain Architecture Design Document

## Overview

This document describes the architecture of an artificial intelligence system that learns cause and effect through prediction and error correction. The brain is fundamentally a **prediction machine** - every neuron and pattern exists to predict what comes next.

## Core Principles

### 1. Prediction is the Foundation
The brain continuously predicts future events and actions. Learning occurs when predictions fail - these failures create structure that captures the context needed to make better predictions.

### 2. Abstraction Emerges from Failure
When a prediction fails, a **pattern** is created to remember the specific context where the failure occurred. Higher levels of abstraction don't exist by design - they emerge because lower levels made mistakes that needed correction.

### 3. Peak Neurons are Decision Points
A **peak neuron** is a neuron that has learned something. It encountered a situation where its simple connections weren't sufficient, so it created a pattern to remember "in THIS context, predict THAT instead."

### 4. Cause and Effect Interweave
Events predict events. Events predict actions. Patterns of events predict patterns. Actions feed back as observations. Two hierarchies dance together:
- **Event hierarchy**: Learns what IS (passive, association-based)
- **Action hierarchy**: Learns what to DO (active, trial-and-error)

### 5. Distributed Cognition via Voting
There's no central controller. Every active neuron and pattern contributes its vote. Intelligence emerges from consensus, weighted by level and temporal proximity.

### 6. Time is Structural
Temporal distance is built into connections. The frame-by-frame processing and distance-based decay make time a first-class citizen, not an afterthought.

---

## Two Hierarchies: Events and Actions

### Event Hierarchy (Passive Learning)

Event neurons observe what happens in the world. They learn by **association** - when events co-occur or follow each other, connections form and strengthen.

#### Inference Mechanisms

**Connection Inference**: Active event neurons predict future events via their connections.
- When prediction comes true → connection naturally strengthens through observation
- When prediction fails with high confidence → create a pattern to correct future predictions

**Pattern Inference**: When a pattern matches the current context, it overrides connection inference.
- Pattern_past defines the context (neurons active when the peak was observed)
- Pattern_future defines the prediction (what the pattern predicts will happen)

#### Pattern Learning for Events

When a pattern is observed (matched to current context):
1. **Pattern_past merges with observation**:
   - Neurons in both pattern and observation → strengthen
   - Neurons only in observation → add with strength 1 (novel context)
   - Neurons only in pattern → weaken (missing context)
2. Pattern definition remains fluid, adapting to what's actually observed

When pattern inference succeeds:
- Pattern_future strength for that event increases

When pattern inference fails:
- Pattern_future adds the actual outcome (novel neurons)
- Pattern_future weakens the failed prediction
- The pattern learns to predict the correct neuron next time

### Action Hierarchy (Active Learning)

Actions are always at the **base level**. All event neurons (at any level) vote on which actions to take. Actions learn by **trial-and-error** with rewards, not by association.

#### Key Differences from Events

| Aspect | Events | Actions |
|--------|--------|---------|
| Learning signal | Strength (observation) | Reward (feedback) |
| Ground truth | What actually happened | None - only reward signals |
| Winner selection | Highest total strength | Highest weighted reward |
| Pattern_future learning | Merge with ground truth | Explore alternatives on pain |

#### Exploration

Exploration kick-starts the search for optimal actions:
- When **no action winners exist** for a channel, explore randomly
- Exploration asks the channel for an unexplored action
- Exploration builds connections between event neurons/patterns and actions

#### Action Pattern Learning

When action pattern inference results in **positive reward**:
- Pattern_future strength for that action increases
- Reward is updated via exponential smoothing

When action pattern inference results in **negative reward** (pain):
- Pattern finds another valid action in the channel
- Adds it to pattern_future with neutral reward
- Continues until all possible actions are in pattern_future
- Over time, reward-weighted selection favors the best-rewarding action

---

## Voting Architecture

### Vote Collection
All active neurons (at all levels) cast votes for what they predict will happen next.

**Connection votes** (from base level neurons):
- Source: active base neurons with connections to target neurons
- Condition: `connection.distance = neuron.age + 1` (predicting next frame)

**Pattern votes** (from pattern neurons at any level):
- Source: active pattern neurons with pattern_future entries
- Condition: `pattern_future.distance = pattern.age + 1` (predicting next frame)

### Vote Weighting

Each vote is weighted by two factors:

1. **Level weight**: `1 + level * levelVoteMultiplier`
   - Higher-level patterns have more influence (they represent more context)
   - Default multiplier: 3x per level

2. **Time decay**: `1 - (distance - 1) * (1 / contextLength)`
   - Recent predictions weighted more than distant ones
   - Distance 1 gets full weight, distance 9 gets ~10% weight

**Effective strength** = `levelWeight * timeDecay * rawStrength`

### Pattern Override Rule
When a peak neuron has both connection votes AND pattern votes active, **pattern votes override connection votes**. This is implemented by deleting connection votes from neurons that are peaks of voting patterns.

### Consensus Determination

1. **Vote aggregation**: Sum effective strengths per target neuron
2. **Reward calculation**: Weighted average of rewards (for actions)
3. **Per-dimension ranking**:
   - Events: highest total strength wins
   - Actions: highest weighted reward wins
4. **Winner selection**: Neurons ranked #1 in any dimension become winners

---

## Data Structures

### Neurons
- **Base neurons (level 0)**: Have coordinates, represent specific observations/actions
- **Pattern neurons (level 1+)**: No coordinates, represent learned contexts
- **Type**: 'event' (observations) or 'action' (decisions)
- **Channel**: Which sensory/motor channel this neuron belongs to

### Connections
- Link base neurons across time (`distance` = temporal gap)
- Store `strength` (how often observed) and `reward` (expected outcome for actions)
- Only event neurons can be sources - actions cannot predict
- Rewards updated via exponential smoothing: `new = smooth * observed + (1 - smooth) * old`

### Patterns
- **pattern_peaks**: Maps pattern neuron to its peak neuron (the decision node)
- **pattern_past**: Context neurons active when peak was observed (with relative ages)
- **pattern_future**: Predictions from the pattern (base neurons with distances)

### Active Memory Tables
- **active_neurons**: Currently active neurons with their ages (sliding window)
- **matched_patterns**: Patterns that matched in current frame
- **matched_pattern_past**: Context analysis for matched patterns (common/novel/missing)
- **inference_votes**: All votes before and after pattern override
- **inferred_neurons**: Final predictions with winner flags

---

## Frame Processing Flow

```
1. processFrameIO()
   - Get frame events from all channels
   - Get previous frame's action winners from inferred_neurons
   - Execute actions in channels
   - Get rewards from channels

2. ageNeurons()
   - Increment age of all active neurons

3. processBaseNeurons()
   - Find/create neurons for frame points
   - Insert as active neurons at age 0
   - Reinforce connections (event→event, event→action)
   - Apply rewards to action connections
   - Track prediction accuracy

4. processPatternNeurons()
   - recognizePatterns(): Match and activate patterns level by level
   - refinePatterns(): Update pattern_future based on observations/rewards
   - learnNewPatterns(): Create patterns from prediction errors and action regret

5. deactivateOldNeurons()
   - Remove neurons that aged out of context window

6. inferNeurons()
   - collectVotes(): Gather connection and pattern votes
   - Delete overridden votes (pattern override rule)
   - Aggregate and rank to determine winners
   - applyExploration(): Add exploration actions if no winners
   - saveInferences(): Store for next frame

7. runForgetCycle() (periodic)
   - Decay connection and pattern strengths
   - Delete zero-strength entries
   - Clean up orphaned pattern neurons
```

---

## Channel Interface

Channels are adapters between the brain and external devices (eyes, ears, trading systems, etc.).

### Required Methods
- `getEventDimensions()`: Input dimension names
- `getOutputDimensions()`: Output dimension names
- `getActionNeurons()`: All possible action coordinates (for pre-creation)
- `getFrameEvents()`: Current observations
- `executeOutputs(actions)`: Execute brain's decisions
- `getRewards()`: Feedback on action outcomes (0 = neutral)
- `initialize()`: Channel-specific setup

### Optional Methods
- `debugVotes(votes, brain)`: Display vote details for debugging
- `onEventPredictions(winners)`: Receive winning event predictions
- `getPredictionMetrics()`: Return continuous prediction error metrics
- `getOutputPerformanceMetrics()`: Return channel-specific performance (e.g., P&L)

---

## Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| contextLength | 5 | Frames a neuron stays active |
| levelVoteMultiplier | 3 | Weight increase per level |
| rewardExpSmooth | 0.9 | Exponential smoothing for rewards |
| eventErrorMinStrength | 2.0 | Min strength to create error pattern |
| actionRegretMinStrength | 2.0 | Min strength to create regret pattern |
| mergePatternThreshold | 0.5 | Min match ratio for pattern recognition |
| patternNegativeReinforcement | 0.1 | Weakening rate for missing context |
| forgetCycles | 100 | Frames between forget cycles |
| connectionForgetRate | 1 | Strength decay per forget cycle |
| patternForgetRate | 1 | Pattern strength decay per forget cycle |
| maxLevels | 10 | Maximum pattern hierarchy depth |

---

## Summary

This architecture implements a theory of how minds work:
- **Prediction** drives all learning
- **Failure** creates structure (patterns)
- **Events** learn passively through association
- **Actions** learn actively through trial-and-error
- **Voting** enables distributed decision-making with level and time weighting
- **Time** is built into the representation
- **Patterns override connections** to correct prediction errors

The code is just the implementation. The architecture is a model of intelligence.

