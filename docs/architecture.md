# Brain Architecture Design Document

## Overview

This document describes the architecture of an artificial intelligence system that learns cause and effect through prediction and error correction. The brain is fundamentally a **prediction machine** - every neuron and pattern exists to predict what comes next.

## Core Principles

### 1. Prediction is the Foundation
The brain continuously predicts future events and actions. Learning occurs when predictions fail - these failures create structure that captures the context needed to make better predictions.

### 2. Abstraction Emerges from Failure
When a prediction fails, a **pattern** is created to remember the specific context where the failure occurred. Higher levels of abstraction don't exist by design - they emerge because lower levels made mistakes that needed correction.

### 3. Parent Neurons are Decision Points
A **parent neuron** is a neuron that has learned something. It encountered a situation where its simple connections weren't sufficient, so it created a pattern to remember "in THIS context, predict THAT instead."

### 4. Cause and Effect Interweave
Events predict events. Events predict actions. Patterns of events predict patterns. Actions feed back as observations. Two hierarchies dance together:
- **Event hierarchy**: Learns what IS (passive, association-based)
- **Action hierarchy**: Learns what to DO (active, trial-and-error)

### 5. Distributed Cognition via Voting
There's no central controller. Every active neuron and pattern contributes its vote. Intelligence emerges from consensus, weighted by level and temporal proximity.

### 6. Time is Structural
Temporal distance is built into connections. The frame-by-frame processing and distance-based decay make time a first-class citizen, not an afterthought.

---

## Implementation Architecture

### Core Classes

The system uses an **in-memory architecture** with optional MySQL persistence:

#### Brain (`brain/brain.js`)
Main orchestrator that coordinates all components:
- Frame processing loop
- Pattern recognition and learning
- Inference and voting
- Forget cycles
- Backup/restore to MySQL

#### Thalamus (`brain/thalamus.js`)
Relay station for reference frame transfers (named after biological thalamus):
- **Neuron registry**: Maps neuron IDs to Neuron objects
- **Neuron lookup**: Fast coordinate-based lookup for sensory neurons
- **Channel management**: Instantiates and coordinates channels
- **Dimension mappings**: Translates dimension names to IDs
- **Action execution**: Coordinates multi-channel action execution

#### Memory (`brain/memory.js`)
Temporal sliding window for short-term memory:
- **Active neurons**: Indexed by age (0 = newest, contextLength-1 = oldest)
- **Inferred neurons**: Winning predictions from previous frame
- **Context retrieval**: Gets context for pattern matching and learning
- **Aging**: Shifts temporal window each frame

#### Neuron (`brain/neuron.js`)
Unified class for all neurons (sensory and pattern):
- **Connections**: Map<distance, Map<toNeuron, {strength, reward}>>
- **Patterns**: Set of pattern neurons (for peak neurons)
- **Context**: Context object (for pattern neurons)
- **Voting**: Generates votes weighted by level and time
- **Learning**: Creates connections and patterns from observations

#### Context (`brain/context.js`)
Pattern context representation and matching:
- **Entries**: Array of {neuron, distance, strength}
- **Keys**: Set for fast O(1) lookup during matching
- **Matching**: Threshold-based pattern recognition
- **Merging**: Strengthens common, adds novel, weakens missing

---

## Two Hierarchies: Events and Actions

### Event Hierarchy (Passive Learning)

Event neurons observe what happens in the world. They learn by **association** - when events co-occur or follow each other, connections form and strengthen.

#### Connection Learning

When a neuron is active at age > 0 and a new neuron appears at age 0:
- Create or strengthen connection at distance = age
- Increment strength (clamped to maxStrength)
- Connections predict: "when I appear, this other neuron appears N frames later"

#### Pattern Learning for Events

When a neuron makes a strong prediction that fails:
- Create pattern with parent = predictor neuron
- Context = all active neurons at time of prediction
- Prediction = actual outcome (not the failed prediction)
- Pattern activates when context matches in future

When a pattern matches and predicts correctly:
- Pattern context strengthens (common neurons)
- Pattern prediction strengthens
- Pattern learns to predict better

When a pattern matches but predicts incorrectly:
- Pattern context adapts (add novel, weaken missing)
- Pattern prediction adapts (add actual, weaken failed)

### Action Hierarchy (Active Learning)

Actions are always at the **base level** (level 0). All neurons (at any level) vote on which actions to take. Actions learn by **trial-and-error** with rewards, not by association.

#### Key Differences from Events

| Aspect | Events | Actions |
|--------|--------|---------|
| Learning signal | Strength (observation) | Reward (feedback) |
| Ground truth | What actually happened | None - only reward signals |
| Winner selection | Highest total strength | Highest weighted reward |
| Connection learning | Strengthen when observed | Update reward via smoothing |
| Pattern learning | Created on prediction error | Created on negative reward (regret) |

#### Exploration

Exploration ensures all channels have actions:
- When **no action inferred** for a channel, use deterministic exploration
- Select lowest-ID action from channel's action set
- Builds connections between active neurons and exploration actions
- Enables learning from random exploration

#### Action Pattern Learning

When action connection/pattern results in **positive reward**:
- Connection/pattern strength increases
- Reward updates via exponential smoothing
- Action becomes more likely in similar contexts

When action connection/pattern results in **negative reward** (regret):
- If strength >= actionRegretMinStrength, create regret pattern
- Pattern learns alternative actions for this context
- Over time, reward-weighted selection favors best action

---

## Voting Architecture

### Vote Collection

All active neurons (at all levels) cast votes for what they predict will happen next.

**Implementation** (`brain.collectVotes()`):
1. Iterate through active neurons at ages 0 to contextLength-2
2. Skip neurons with activated patterns (pattern override)
3. Call `neuron.vote(age, timeDecay)` to get votes
4. Save votes and context in memory for pattern learning
5. Return all votes for consensus

**Vote Generation** (`neuron.vote()`):
- Get connections at distance = age + 1 (predicting next frame)
- Weight each connection by level and time
- Return array of {neuron, strength, reward, distance}

### Vote Weighting

Each vote is weighted by two factors:

1. **Level weight**: `1 + level * levelVoteMultiplier`
   - Higher-level patterns have more influence (they represent more context)
   - Default multiplier: 3x per level
   - Example: level 0 = 1x, level 1 = 4x, level 2 = 7x

2. **Time decay**: `1 - age / contextLength`
   - Recent predictions weighted more than distant ones
   - Age 0 gets full weight (1.0), age 4 gets 20% weight (0.2) with contextLength=5

**Effective strength** = `levelWeight * timeWeight * rawStrength`

### Pattern Override Rule

When a pattern activates on a parent neuron, the parent's connection votes are suppressed.

**Implementation**:
- During pattern recognition, matched patterns are activated
- `memory.activatePattern(pattern, parent, age)` sets `state.activatedPattern = pattern`
- During vote collection, neurons with `state.activatedPattern !== null` are skipped
- This prevents the parent from voting via its connections when a pattern is active

**Why**: Patterns exist to correct connection predictions. When a pattern matches, it knows better than the raw connections.

### Consensus Determination

**Implementation** (`brain.determineConsensus()`):

1. **Aggregate votes**: Sum effective strengths per target neuron
2. **Calculate rewards**: Weighted average of vote rewards (for actions)
3. **Select winners per dimension**:
   - Events: highest total strength wins
   - Actions: highest weighted reward wins
4. **Return winners**: Neurons that won in any dimension

**Example**:
```
Votes: [
  {neuron: price_up, strength: 10, reward: 0.5},
  {neuron: price_up, strength: 5, reward: 0.3},
  {neuron: price_down, strength: 8, reward: -0.2}
]

Aggregation:
  price_up: strength=15, reward=(10*0.5 + 5*0.3)/15 = 0.43
  price_down: strength=8, reward=-0.2

Winner (event): price_up (highest strength)
Winner (action): price_up (highest reward)
```

---

## Data Structures

### In-Memory Structures

#### Neuron Class
```javascript
class Neuron {
  id: number                    // Unique ID
  level: number                 // 0 = sensory, 1+ = pattern

  // Sensory neurons (level 0)
  channel: string               // Channel name
  type: 'event' | 'action'      // Neuron type
  coordinates: object           // {dimension: value}

  // Pattern neurons (level > 0)
  peak: Neuron                  // Peak neuron reference

  // All neurons
  connections: Map<distance, Map<Neuron, {strength, reward}>>
  patterns: Set<Neuron>         // Pattern neurons (for peaks)
  context: Context              // Pattern context (for patterns)
  contextRefs: Map<Neuron, Set<distance>>  // Reverse references
  activationStrength: number    // Incremented on activation
}
```

#### Memory Class
```javascript
class Memory {
  activeNeurons: Array<Map<Neuron, {activatedPattern, votes, context}>>
  inferredNeurons: Array<{neuron, strength, reward}>
  contextLength: number
}
```

#### Context Class
```javascript
class Context {
  entries: Array<{neuron, distance, strength}>
  keys: Set<string>  // For fast O(1) lookup
}
```

### MySQL Persistence (Optional)

Used for backup/restore between episodes, not during frame processing:

- **`channels`** - Channel registry with IDs
- **`dimensions`** - Dimension names with IDs
- **`neurons`** - All neurons with level
- **`base_neurons`** - Sensory neuron metadata (channel, type)
- **`coordinates`** - Sensory neuron coordinate values
- **`connections`** - Base neuron connections (distance, strength, reward)
- **`pattern_peaks`** - Pattern-to-peak mappings with strength
- **`pattern_past`** - Pattern contexts (context neurons with ages and strengths)

---

## Frame Processing Flow

The brain processes each frame through a coordinated sequence:

### 1. getFrame()
**Purpose**: Collect sensory inputs and previous actions

```javascript
// Get events from all channels
for (channel of channels) {
  events = channel.getFrameEvents()
  frame.push({coordinates, channel, type: 'event'})
}

// Get actions from previous inference
actions = memory.getInferredActions()
for (action of actions) {
  frame.push({coordinates, channel, type: 'action'})
}
```

### 2. getRewards()
**Purpose**: Get feedback on executed actions

```javascript
for (channel of channels) {
  reward = channel.getRewards(actions)
  if (reward !== 0) rewards.set(channel, reward)
}
```

### 3. memory.age()
**Purpose**: Shift temporal window

```javascript
// Shift ages: 0→1, 1→2, ..., contextLength-1→deleted
activeNeurons.unshift(new Map())
if (activeNeurons.length > contextLength) {
  removed = activeNeurons.pop()  // Deactivate aged-out neurons
}
```

### 4. activateSensors()
**Purpose**: Activate sensory neurons for current frame

```javascript
// Find or create neurons for frame points
neurons = getFrameNeurons(frame)  // via Thalamus

// Activate at age 0
for (neuron of neurons) {
  memory.activateNeuron(neuron)
  neuron.strengthenActivation()
}

// Track inference accuracy
diagnostics.trackInferencePerformance(...)
```

### 5. recognizePatterns()
**Purpose**: Detect and activate patterns hierarchically

```javascript
level = 0
while (true) {
  newNeurons = memory.getNewNeurons(level)
  if (newNeurons.length === 0) break

  context = new Context()
  for ({neuron, age} of memory.getContextNeurons(level))
    context.addNeuron(neuron, age, 1)

  // Match patterns for each parent
  for (parent of newNeurons) {
    pattern = parent.matchPattern(context)
    if (pattern) memory.activatePattern(pattern, parent, 0)
  }

  level++
  if (level >= maxLevels) break
}
```

### 6. updateConnections()
**Purpose**: Learn connections from observations

```javascript
newActiveNeurons = memory.getNewSensoryNeurons()  // age=0, level=0

for ({neuron, age} of memory.getContextNeurons()) {
  neuron.learnConnections(age, newActiveNeurons, rewards, channelActions)
}
```

**Neuron.learnConnections()**:
```javascript
distance = age
for (newNeuron of newActiveNeurons) {
  if (hasConnection(distance, newNeuron)) {
    updateConnection(distance, newNeuron, reward)  // increment strength, smooth reward
  } else {
    createConnection(distance, newNeuron, 1, reward)
  }
}
```

### 7. learnNewPatterns()
**Purpose**: Create patterns from prediction errors and regret

```javascript
newActiveNeurons = memory.getNewSensoryNeurons()

for ({neuron, age, votes, context} of memory.getVotersWithContext()) {
  newPattern = neuron.learnNewPattern(age, context, votes, newActiveNeurons, rewards, channelActions)
  if (newPattern) {
    thalamus.addNeuron(newPattern)
    memory.activatePattern(newPattern, neuron, age)  // neuron becomes parent
  }
}
```

**Neuron.learnNewPattern()**:
```javascript
// Check for prediction errors (events)
for (vote of votes) {
  if (vote.neuron.type === 'event' && vote.strength >= eventErrorMinStrength) {
    if (!newActiveNeurons.has(vote.neuron)) {
      // Strong prediction failed - create error pattern
      return createPattern(context, newActiveNeurons)
    }
  }
}

// Check for action regret
for (vote of votes) {
  if (vote.neuron.type === 'action' && vote.strength >= actionRegretMinStrength) {
    reward = rewards.get(vote.neuron.channel)
    if (reward < actionRegretMinPain) {
      // Painful action - create regret pattern
      return createPattern(context, alternativeActions)
    }
  }
}
```

### 8. inferNeurons()
**Purpose**: Predict next frame via voting

```javascript
// Collect votes from active neurons
votes = collectVotes()

// Determine consensus
inferences = determineConsensus(votes)

// Ensure all channels have actions
ensureChannelActions(inferences)

// Save for next frame
memory.saveInferences(inferences)
```

### 9. executeActions()
**Purpose**: Execute inferred actions

```javascript
channelActions = memory.getInferredActions()
thalamus.executeChannelActions(channelActions)
```

### 10. runForgetCycle() (periodic)
**Purpose**: Prevent curse of dimensionality

```javascript
if (frameNumber % forgetCycles !== 0) return

// Forget connections and patterns
deadPatterns = thalamus.forgetNeurons()

// Delete dead patterns recursively
deletePatterns(deadPatterns)
```

**Neuron.forget()**:
```javascript
// Decay connections
for (distanceMap of connections.values()) {
  for (connection of distanceMap.values()) {
    connection.strength -= connectionForgetRate
    if (connection.strength <= 0) delete connection
  }
}

// Decay pattern context
for (entry of context.entries) {
  entry.strength -= contextForgetRate
  if (entry.strength <= 0) remove entry
}

// Decay pattern predictions
// (stored in pattern neurons, not shown here)

// Return true if pattern can be deleted (no content, no references)
return canDelete()
```

---

## Channel Interface

Channels are adapters between the brain and external devices (eyes, ears, trading systems, etc.).

### Required Methods

```javascript
class Channel {
  // Define coordinate space
  getEventDimensions()    // Returns: Array<Dimension>
  getActionDimensions()   // Returns: Array<Dimension>

  // Pre-create action neurons
  getActions()            // Returns: Array<coordinates>

  // Frame processing
  getFrameEvents()        // Returns: Array<coordinates>
  executeOutputs(actions) // Execute brain's decisions
  getRewards(actions)     // Returns: number (0 = neutral, + = good, - = bad)
}
```

### Optional Methods

```javascript
class Channel {
  // Initialization
  static initialize(options)           // Channel-level setup
  static resetChannelContext()         // Reset shared state

  // Coordinated execution
  static executeChannelActions(channels, actionsMap)  // Multi-channel coordination
  static getPortfolioMetrics(channels)                // Aggregate metrics

  // Instance methods
  resetContext()                       // Reset instance state
  calculatePredictionError()           // Continuous error (e.g., MAPE)
  getOutputPerformanceMetrics()        // Channel performance (e.g., P&L)
  getMetrics()                         // Diagnostic metrics
}
```

### Example: Stock Channel

```javascript
class StockChannel extends Channel {
  getEventDimensions() {
    return [
      new Dimension('price_change', this.id, 'event'),
      new Dimension('volume_change', this.id, 'event'),
      new Dimension('position', this.id, 'event')
    ]
  }

  getActionDimensions() {
    return [new Dimension('action', this.id, 'action')]
  }

  getActions() {
    return [
      {action: 'buy'},
      {action: 'sell'},
      {action: 'hold'}
    ]
  }

  async getFrameEvents() {
    // Return current price/volume changes and position
    return [{
      price_change: this.discretize(priceChange),
      volume_change: this.discretize(volumeChange),
      position: this.position
    }]
  }

  async executeOutputs(actions) {
    // Execute trade decision
    if (actions[0].coordinates.action === 'buy') this.position = 1
    else if (actions[0].coordinates.action === 'sell') this.position = -1
  }

  async getRewards(actions) {
    // Return profit/loss from trade
    return this.position * this.priceChange
  }
}
```

---

## Key Hyperparameters

Configured in `Neuron`, `Context`, `Memory`, and `Brain` classes:

| Parameter | Default | Location | Description |
|-----------|---------|----------|-------------|
| contextLength | 5 | Memory | Frames a neuron stays active |
| maxStrength | 100 | Neuron/Context | Maximum connection/pattern strength |
| minStrength | 0 | Neuron/Context | Minimum strength before deletion |
| levelVoteMultiplier | 3 | Neuron | Weight increase per pattern level |
| rewardSmoothing | 1 | Neuron | Exponential smoothing for rewards (1 = full replacement) |
| eventErrorMinStrength | 2 | Neuron | Min strength to create error pattern |
| actionRegretMinStrength | 2 | Neuron | Min strength to create regret pattern |
| actionRegretMinPain | 0 | Neuron | Min negative reward to trigger regret |
| mergeThreshold | 0.5 | Context | Min match ratio for pattern recognition |
| negativeReinforcement | 0.1 | Context | Weakening rate for missing context |
| connectionForgetRate | 1 | Neuron | Connection strength decay per forget cycle |
| contextForgetRate | 1 | Neuron | Pattern context strength decay per forget cycle |
| patternForgetRate | 1 | Neuron | Pattern prediction strength decay per forget cycle |
| forgetCycles | 100 | Brain | Frames between forget cycles |
| maxLevels | 10 | Brain | Maximum pattern hierarchy depth |

---

## Summary

This architecture implements a theory of how minds work:

### Core Principles
- **Prediction** drives all learning
- **Failure** creates structure (patterns)
- **Events** learn passively through association
- **Actions** learn actively through trial-and-error
- **Voting** enables distributed decision-making with level and time weighting
- **Time** is built into the representation
- **Patterns override connections** to correct prediction errors

### Implementation Highlights
- **In-memory processing**: All learning in JavaScript objects (no DB queries during frames)
- **Unified neuron class**: Sensory and pattern neurons share common functionality
- **Thalamus relay**: Centralizes neuron registry and channel coordination
- **Temporal sliding window**: Memory manages active neurons by age
- **Context matching**: Fast threshold-based pattern recognition
- **Optional persistence**: MySQL backup/restore between episodes

The code is just the implementation. The architecture is a model of intelligence.

