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
There's no central controller. Every active neuron and pattern contributes its vote. Intelligence emerges from consensus, weighted by level and confidence.

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
- Pattern_past defines the context (what connections led TO the peak)
- Pattern_future defines the prediction (what the pattern predicts will happen)

#### Pattern Learning for Events

When a pattern is observed (matched to current context):
1. **Pattern_past merges with observation**:
   - Connections in both pattern and observation → strengthen
   - Connections only in observation → add with strength 1
   - Connections only in pattern → weaken (eventually delete)
2. Pattern definition remains fluid, adapting to what's actually observed

When pattern inference succeeds:
- Pattern_future strength for that event increases

When pattern inference fails:
- Pattern_future merges with what actually happened (same merge logic as pattern_past)
- The pattern learns to predict the correct neuron next time

### Action Hierarchy (Active Learning)

Actions are always at the **base level**. All event neurons (at any level) vote on which actions to take. Actions learn by **trial-and-error** with rewards, not by association.

#### Key Differences from Events

| Aspect | Events | Actions |
|--------|--------|---------|
| Learning signal | Strength (observation) | Reward (feedback) |
| Ground truth | What actually happened | None - only reward signals |
| Vote calculation | `strength * level_weight` | `reward * level_weight` |
| Pattern_future learning | Merge with ground truth | Explore alternatives |

#### Exploration

Exploration kick-starts the search for optimal actions:
- When no action neurons are inferred, neurons explore randomly
- As inference confidence grows, exploration decreases (but never stops completely)
- Exploration builds connections between event neurons/patterns and actions

#### Action Pattern Learning

When action pattern inference results in **positive reward**:
- Pattern_future strength for that action increases

When action pattern inference results in **negative reward**:
- Pattern finds another valid action in the channel
- Adds it to pattern_future with zero reward
- Continues until all possible actions are in pattern_future
- Over time, Boltzmann selection favors the best-rewarding action

---

## Voting Architecture

### Vote Collection
All active neurons (at all levels) cast votes for what they predict will happen next.

### Consensus Determination
Simple strength-based voting - no weighting schemes:

1. **Vote aggregation**: Each source neuron votes once per target
   - Sum all vote strengths equally (no distance or level weighting)
   - Neurons at the right distance for cyclic patterns naturally have stronger connections
   - Let the learned connection strengths speak for themselves

2. **Per-dimension selection**: Each dimension selects one winner
   - For events: highest total strength wins (deterministic)
   - For actions: Boltzmann selection based on total reward (probabilistic)

### Pattern Override Rule
When a peak neuron has both connection inferences AND pattern inferences active, **pattern always wins**. The purpose of patterns is to correct connection inference - thresholds only apply during initial pattern creation.

---

## Data Structures

### Neurons
- **Base neurons (level 0)**: Have coordinates, represent specific observations/actions
- **Pattern neurons (level 1+)**: No coordinates, represent learned contexts
- **Type**: 'event' (observations) or 'action' (decisions)
- **Channel**: Which sensory/motor channel this neuron belongs to

### Connections
- Link neurons across time (`distance` = temporal gap)
- Store `strength` (how often observed) and `reward` (expected outcome for actions)
- Cross-level connections allowed: same-level and high→base (not base→high)
- **Action neurons can NEVER be sources** - only events can predict

### Patterns
- **pattern_peaks**: Maps pattern neuron to its peak neuron (the decision node)
- **pattern_past**: Connections leading TO the peak (defines the context)
- **pattern_future**: Connections FROM the peak (defines the prediction)

---

## Frame Processing Flow

```
1. Age neurons and connections (slide temporal window)
2. Recognize neurons from frame input
   - Activate base neurons
   - Match and activate patterns (recursively up levels)
3. Populate inference sources for executed actions
4. Learn from base level
   - Validate event predictions (accuracy tracking)
   - Negative reinforcement for failed event predictions
   - Apply rewards to action predictions
5. Learn from inference level
   - Merge pattern_future with observations
   - Create error patterns from failed predictions
6. Infer neurons (voting)
   - Collect votes from all levels
   - Determine consensus per dimension
   - Apply exploration to actions
   - Save inferences for next frame
7. Run forget cycle (periodic cleanup)
```

---

## Channel Interface

Channels are adapters between the brain and external devices (eyes, ears, trading systems, etc.).

### Required Methods
- `getEventDimensions()`: Input dimension names
- `getOutputDimensions()`: Output dimension names
- `getFrameEvents()`: Current observations
- `executeOutputs(actions)`: Execute brain's decisions
- `getRewards()`: Feedback on action outcomes
- `getExplorationAction(votedActions)`: Provide unexplored action for trial

---

## Summary

This architecture implements a theory of how minds work:
- **Prediction** drives all learning
- **Failure** creates structure (patterns)
- **Events** learn passively through association
- **Actions** learn actively through trial-and-error
- **Voting** enables distributed decision-making
- **Time** is built into the representation

The code is just the implementation. The architecture is a model of intelligence.

